import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import imagetransforms

import loggy

logger = loggy.setup_custom_logger('root', "train_cnn_lstm.py")

from warpctc_pytorch import CTCLoss
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import time
import shutil

from madcat import MadcatDataset
from iam import IAMDataset
from rimes import RimesDataset
from datautils import GroupedSampler, SortByWidthCollater
from models.cnnlstm import CnnOcrModel
from textutils import *
import argparse


from english import EnglishAlphabet

from sklearn.metrics import roc_auc_score

from lr_scheduler import ReduceLROnPlateau
#from tensorboard import SummaryWriter


def test_on_val(val_dataloader, model, criterion):
    start_val = time.time()
    cer_running_avg = 0
    wer_running_avg = 0
    cer_lm_running_avg = 0
    wer_lm_running_avg = 0
    loss_running_avg = 0
    n_samples = 0

    display_hyp = True

    # To start, put model in eval mode
    model.eval()

    logger.info("About to comptue %d val batches" % len(val_dataloader))
    for input_tensor, target, input_widths, target_widths, metadata in val_dataloader:
        # In validation set, not doing backprop, so set volatile to True to reduce memory footprint
        input_tensor = Variable(input_tensor.cuda(async=True), volatile=True)
        target = Variable(target, volatile=True)
        target_widths = Variable(target_widths, volatile=True)
        input_widths = Variable(input_widths, volatile=True)

        model_output, model_output_actual_lengths = model(input_tensor, input_widths)
        loss = criterion(model_output, target, model_output_actual_lengths, target_widths)

        hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)
        hyp_transcriptions_lm = hyp_transcriptions
        # hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=True)

        batch_size = input_tensor.size(0)
        curr_loss = loss.data[0] / batch_size
        n_samples += 1
        loss_running_avg += (curr_loss - loss_running_avg) / n_samples

        cur_target_offset = 0
        batch_cer = 0
        batch_wer = 0
        batch_cer_lm = 0
        batch_wer_lm = 0
        target_np = target.data.numpy()
        ref_transcriptions = []
        for i in range(len(hyp_transcriptions)):
            ref_transcription = form_target_transcription(
                target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])],
                model.alphabet
            )
            ref_transcriptions.append(uxxxx_to_utf8(ref_transcription))
            cur_target_offset += target_widths.data[i]
            cer, wer = compute_cer_wer(hyp_transcriptions[i], ref_transcription)

            batch_cer += cer
            batch_wer += wer

            cer_lm, wer_lm = compute_cer_wer(hyp_transcriptions_lm[i], ref_transcription)
            batch_cer_lm += cer_lm
            batch_wer_lm += wer_lm

        cer_running_avg += (batch_cer / batch_size - cer_running_avg) / n_samples
        wer_running_avg += (batch_wer / batch_size - wer_running_avg) / n_samples

        cer_lm_running_avg += (batch_cer_lm / batch_size - cer_lm_running_avg) / n_samples
        wer_lm_running_avg += (batch_wer_lm / batch_size - wer_lm_running_avg) / n_samples

        # For now let's display one set of transcriptions every test, just to see improvements
        if display_hyp:
            logger.info("--------------------")
            logger.info("Sample hypothesis / reference transcripts")
            logger.info("Error rate for this batch is:\tNo LM CER: %f\tWER:%f" % (
            batch_cer / batch_size, batch_wer / batch_size))
            logger.info("\t\tWith LM CER: %f\tWER: %f" % (batch_cer_lm / batch_size, batch_wer_lm / batch_size))
            logger.info("")
            hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)
            # hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=False)
            hyp_transcriptions_lm = hyp_transcriptions
            for i in range(len(hyp_transcriptions)):
                logger.info("\tHyp[%d]: %s" % (i, hyp_transcriptions[i]))
                logger.info("\tHyp-LM[%d]: %s" % (i, hyp_transcriptions_lm[i]))
                logger.info("\tRef[%d]: %s" % (i, ref_transcriptions[i]))
                logger.info("")
            logger.info("--------------------")
            display_hyp = False

    # Finally, put model back in train mode
    model.train()
    end_val = time.time()
    logger.info("Total val time: %s" % (end_val - start_val))
    return loss_running_avg, cer_running_avg, wer_running_avg, cer_lm_running_avg, wer_lm_running_avg


def train(batch, model, criterion, optimizer):
    input_tensor, target, input_widths, target_widths, metadata = batch
    input_tensor = Variable(input_tensor.cuda(async=True))
    target = Variable(target)
    target_widths = Variable(target_widths)
    input_widths = Variable(input_widths)
    optimizer.zero_grad()
    model_output, model_output_actual_lengths = model(input_tensor, input_widths)
    loss = criterion(model_output, target, model_output_actual_lengths, target_widths)
    loss.backward()
    
    # RNN Backprop can have exploding gradients (even with LSTM), so make sure
    # we clamp the abs magnitude of individual gradient entries
    for param in model.parameters():
        if not param.grad is None:
            param.grad.data.clamp_(min=-5, max=5)

    # Okay, now we're ready to update parameters!
    optimizer.step()
    return loss.data[0]



def plot_lr_schedule(wer_array, cer_array, loss_array, lr_points, iteration_points, iteration,
                     snapshot_every_n_iterations, filename='train_cnn_lstm', file_path=os.environ.get('TMPDIR')):
    """
    A method to visualize the LR update from ReduceLROnPlateau.
    More detailed plots for Loss, WER and CER can be obtained from tensorboard files.
    """
    filename = filename.split('/')[-1]
    plt.cla()
    plt.plot(iteration_points, wer_array)
    plt.vlines(lr_points, 0, 1.0)
    plt.savefig(os.path.join(file_path, 'val_lr_wer_%s.png' % (filename)))
    plt.cla()
    plt.plot(iteration_points, cer_array)
    plt.vlines(lr_points, 0, 1.0)
    plt.savefig(os.path.join(file_path, 'val_lr_cer_%s.png' % (filename)))
    plt.cla()
    plt.plot(range(0, iteration), loss_array)
    plt.vlines([p * snapshot_every_n_iterations for p in lr_points], 0, 100.0)
    plt.savefig(os.path.join(file_path, 'val_lr_loss_%s.png' % (filename)))
    plt.cla()


def get_args():
    parser = argparse.ArgumentParser(description="OCR Training Script")
    parser.add_argument("--batch-size", type=int, default=64, help="SGD mini-batch size")
    parser.add_argument("--line-height", type=int, default=30, help="Input image line height")
    parser.add_argument("--hpad", type=int, default=0,
                        help="Amount of horizontal padding to apply to left/right of input image (after resize)")
    parser.add_argument("--vpad", type=int, default=0,
                        help="Amount of vertical padding to apply to top/bottom of input image (after resize)")
    parser.add_argument("--dataset", type=str, required=True, help="Specify dataset, e.g. madcat, iam")
    parser.add_argument("--datadir", type=str, required=True, help="specify the location to unzipped images.")
    parser.add_argument("--snapshot-prefix", type=str, required=True,
                        help="Output directory and basename prefix for output model snapshots")
    parser.add_argument("--load-from-snapshot", type=str,
                        help="Path to snapshot from which we should initialize model weights")
    parser.add_argument("--num-lstm-layers", type=int, required=True, help="Number of LSTM layers in model")
    parser.add_argument("--num-lstm-units", type=int, required=True,
                        help="Number of LSTM hidden units in each LSTM layer (single number, or comma seperated list)")
    parser.add_argument("--lstm-input-dim", type=int, required=True, help="Input dimension for LSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--nepochs", type=int, default=250, help="Maximum number of epochs to train")
    parser.add_argument("--snapshot-num-iterations", type=int, default=2000, help="Every N iterations snapshot model")
    parser.add_argument("--patience", type=int, default=10, help="Patience parameter for ReduceLROnPlateau.")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau")
    parser.add_argument("--tensorboard-log-path", type=str, default="/nfs/isicvlnas01/users/jmathai/logs/tensorboard_logs",
                        help="Path to tensorboard logs")
    parser.add_argument("--plots-path", type=str, default="/nfs/isicvlnas01/users/jmathai/logs/plots",
                        help="Path to plot reduce LR on plateau policy")
    parser.add_argument("--lm-path", type=str, default=None,
                        help="Path to LM directory for decoding(should contain TLG.fst, words.txt, units.txt)")
    parser.add_argument("--acoustic-weight", type=float, default=0.9, help="acoustic weight for the LM")
    return parser.parse_args()


def main():
    logger.info("Starting training\n\n")
    sys.stdout.flush()
    args = get_args()
    snapshot_path = args.snapshot_prefix + "-cur_snapshot.pth"
    best_model_path = args.snapshot_prefix + "-best_model.pth"
    if not args.lm_path is None:
        lm_fst = os.path.join(args.lm_path, 'TLG.fst')
        lm_words = os.path.join(args.lm_path, 'words.txt')
        lm_units = os.path.join(args.lm_path, 'units.txt')


    line_img_transforms = imagetransforms.Compose([
        imagetransforms.Scale(new_h=args.line_height),
        imagetransforms.InvertBlackWhite(),
        imagetransforms.Pad(args.hpad, args.vpad),
        imagetransforms.ToTensor(),
    ])


    if args.dataset.upper() == "IAM":
        fake = 'fake'
        alphabet = EnglishAlphabet(lm_units_path=lm_units) # load the units for LM decoding
        datadir = os.environ.get('TMPDIR') if args.datadir is None else args.datadir
        train_dataset = IAMDataset(datadir, "train", alphabet, args.line_height, line_img_transforms)
        validation_dataset = IAMDataset(datadir, "validation", alphabet, args.line_height, line_img_transforms)
    elif args.dataset.upper() == "RIMES":
        datadir = os.environ.get('TMPDIR') if args.datadir is None else args.datadir
        train_dataset = RimesDataset(datadir, "train", args.line_height, line_img_transforms)
        validation_dataset = RimesDataset(datadir, "validation", args.line_height, line_img_transforms)
    elif args.dataset.upper() == "MADCAT":
        datadir = os.environ.get('TMPDIR') if args.datadir is None else args.datadir
        train_dataset = MadcatDataset(datadir, "train", args.line_height, line_img_transforms)
        validation_dataset = MadcatDataset(datadir, "validation", args.line_height, line_img_transforms)
    else:
        logger.info("Unknown Dataset: %s" % args.dataset.upper())
        sys.exit(1)

    train_dataloader = DataLoader(train_dataset,
                                  args.batch_size,
                                  num_workers=4,
                                  sampler=GroupedSampler(train_dataset, rand=True),
                                  collate_fn=SortByWidthCollater,
                                  pin_memory=True,
                                  drop_last=True)

    validation_dataloader = DataLoader(validation_dataset,
                                       args.batch_size,
                                       num_workers=0,
                                       sampler=GroupedSampler(validation_dataset, rand=True),
                                       collate_fn=SortByWidthCollater,
                                       pin_memory=False,
                                       drop_last=False)


    n_epochs = args.nepochs
    lr_alpha = args.lr
    snapshot_every_n_iterations = args.snapshot_num_iterations

    if not args.load_from_snapshot is None:
        model = CnnOcrModel.FromSavedWeights(args.load_from_snapshot)
    else:
        model = CnnOcrModel(
            num_in_channels=1,
            input_line_height=args.line_height + 2 * args.vpad,
            lstm_input_dim=args.lstm_input_dim,
            num_lstm_layers=args.num_lstm_layers,
            num_lstm_hidden_units=args.num_lstm_units,
            p_lstm_dropout=0.5,
            alphabet=train_dataset.idx_to_char,
            multigpu=True)

    # Set training mode on all sub-modules
    model.train()

    # Currently LM decoding is slow so don't do during training
    #    model.init_lm(lm_fst,
    #                  lm_words,
    #                  acoustic_weight=args.acoustic_weight)

    ctc_loss = CTCLoss().cuda()

    iteration = 0
    best_val_wer_lm = float('inf')

    # Try weight decy of 10^-5??
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_alpha)

    # Maybe try SGD for cnn portion and only Adam for LSTM?
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr_alpha, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, min_lr=args.min_lr)
    wer_array = []
    cer_array = []
    loss_array = []
    lr_points = []
    iteration_points = []

    epoch_size = len(train_dataloader)
    #tensorboard_log_dir = 'tboard-' + args.snapshot_prefix.split('/')[-1] + "_" + datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    #writer = SummaryWriter(os.path.join(args.tensorboard_log_path, tensorboard_log_dir))

    for epoch in range(1, n_epochs + 1):
        epoch_start = datetime.datetime.now()

        # First modify main OCR model
        for batch in train_dataloader:
            sys.stdout.flush()
            iteration += 1
            iteration_start = datetime.datetime.now()


            loss = train(batch, model, ctc_loss, optimizer)

            elapsed_time = datetime.datetime.now() - iteration_start
            loss = loss / args.batch_size

            loss_array.append(loss)

            logger.info("Iteration: %d (%d/%d in epoch %d)\tLoss: %f\tElapsed Time: %s" % (
            iteration, iteration % epoch_size, epoch_size, epoch, loss, pretty_print_timespan(elapsed_time)))

            # Do something with loss, running average, plot to some backend server, etc

            if iteration % snapshot_every_n_iterations == 0:
                logger.info("Testing on validation set")
                val_loss, val_cer, val_wer, val_cer_lm, val_wer_lm = test_on_val(validation_dataloader, model, ctc_loss)
                # Reduce learning rate on plateau
                early_exit = False
                lowered_lr = False
                if scheduler.step(val_wer):
                    lowered_lr = True
                    lr_points.append(iteration / snapshot_every_n_iterations)
                    if scheduler.finished:
                        early_exit = True

                    # for bookeeping only
                    lr_alpha = max(lr_alpha * scheduler.factor, scheduler.min_lr)

                logger.info("Val Loss: %f\tNo LM Val CER: %f\tNo LM Val WER: %f" % (val_loss, val_cer, val_wer))
                logger.info("\t\tWith LM Val CER: %f\tWith LM Val WER: %f" % (val_cer_lm, val_wer_lm))

                torch.save({'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_hyper_params': model.get_hyper_params(),
                            'cur_lr': lr_alpha,
                            'val_loss': val_loss,
                            'val_cer': val_cer,
                            'val_wer': val_wer,
                            'val_cer_lm': val_cer_lm,
                            'val_wer_lm': val_wer_lm},
                           snapshot_path)

                # plotting lr_change on wer, cer and loss.
                wer_array.append(val_wer)
                cer_array.append(val_cer)
                iteration_points.append(iteration / snapshot_every_n_iterations)
                plot_lr_schedule(wer_array, cer_array, loss_array, lr_points,
                                 iteration_points, iteration, snapshot_every_n_iterations,
                                 filename=args.snapshot_prefix, file_path=args.plots_path)

                if val_wer_lm < best_val_wer_lm:
                    logger.info("Best model so far, copying snapshot to best model file")
                    best_val_wer_lm = val_wer_lm
                    shutil.copyfile(snapshot_path, best_model_path)

                logger.info("Running WER: %s" % str(wer_array))
                logger.info("Done with validation, moving on.")

                if early_exit:
                    logger.info("Early exit")
                    sys.exit(0)

                if lowered_lr:
                    logger.info("Switching to best model parameters before continuing with lower LR")
                    weights = torch.load(best_model_path)
                    model.load_state_dict(weights['state_dict'])


        elapsed_time = datetime.datetime.now() - epoch_start
        logger.info("\n------------------")
        logger.info("Done with epoch, elapsed time = %s" % pretty_print_timespan(elapsed_time))
        logger.info("------------------\n")


    #writer.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
