# Script for running full system for OBQA dataset
#!/usr/bin/env bash

set -x
ORIG_DATA_DIR=$1
PREPROCESSED_DATA_DIR=$2
PATH_DIR=$3
PARAM_FILE=$4
FINAL_PATH_DUMPDIR="data/datasets/OBQA/adjusted"
mkdir -p $FINAL_PATH_DUMPDIR

# Tokenization/tagging etc
scripts/preprocess_obqa.sh $ORIG_DATA_DIR $PREPROCESSED_DATA_DIR

# break the train data
scripts/break_train_data_obqa.py $PREPROCESSED_DATA_DIR ${PREPROCESSED_DATA_DIR}/train-split

# path extraction
scripts/path_finder_obqa.sh $PREPROCESSED_DATA_DIR $PATH_DIR

# path adjustments
scripts/path_adjustments_obqa.sh $PREPROCESSED_DATA_DIR $PATH_DIR $FINAL_PATH_DUMPDIR

# Training
MODELDIR="models/obqa"
if [ -d $MODELDIR ]
then
    rm -r $MODELDIR
fi
mkdir -p $MODELDIR
allennlp train --file-friendly-logging -s $MODELDIR --include-package pathnet $PARAM_FILE