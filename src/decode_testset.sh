#!/bin/bash

source ~/.bashrc

# get datapath from argv
model_path=/nfs/isicvlnas01/users/srawls/ocr-dev/tmp-fake-model-best_model.pth
datadir=/nfs/isicvlnas01/users/srawls/ocr-dev/data/iam
lmpath=/nfs/isicvlnas01/users/jmathai/experiments/iam_lm_augment_more_data/IAM-LM/
export PYTHONPATH=/nas/home/srawls/ocr/PyTorchOCR/eesen/:$PYTHONPATH

#cd $TMPDIR
#tar xf ${data_path}


# Run Decoding
python ~/ocr/FlexOCR/src/decode_testset.py --datadir=${datadir}\
                         --model-path=${model_path} \
                         --lm-path=${lmpath}

if [[ $? -ne 0 ]]; then
    exit 1
fi


# Turn decoding output into tokenized words
python ~/ocr/FlexOCR/src/chars_to_tokenized_words.py $TMPDIR/hyp-chars.txt $TMPDIR/hyp-words.txt
python ~/ocr/FlexOCR/src/chars_to_tokenized_words.py $TMPDIR/hyp-lm-chars.txt $TMPDIR/hyp-lm-words.txt
python /nas/home/srawls/ocr/PyTorchOCR/chars_to_tokenized_words.py $TMPDIR/ref-chars.txt $TMPDIR/ref-words.txt


# Do CER measurement
/nfs/isicvlnas01/share/sclite/sclite -r $TMPDIR/ref-chars.txt -h $TMPDIR/hyp-chars.txt -i swb -o all
/nfs/isicvlnas01/share/sclite/sclite -r $TMPDIR/ref-chars.txt -h $TMPDIR/hyp-lm-chars.txt -i swb -o all

# Do WER measurement
/nfs/isicvlnas01/share/sclite/sclite -r $TMPDIR/ref-words.txt -h $TMPDIR/hyp-words.txt -i swb -o all
/nfs/isicvlnas01/share/sclite/sclite -r $TMPDIR/ref-words.txt -h $TMPDIR/hyp-lm-words.txt -i swb -o all


# Now display results
echo "No LM CER:"
grep 'Sum/Avg' $TMPDIR/hyp-chars.txt.sys
echo "LM CER:"
grep 'Sum/Avg' $TMPDIR/hyp-lm-chars.txt.sys

echo "No LM WER:"
grep 'Sum/Avg' $TMPDIR/hyp-words.txt.sys
echo "LM WER:"
grep 'Sum/Avg' $TMPDIR/hyp-lm-words.txt.sys

