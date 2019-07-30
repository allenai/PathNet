# Preprocessing + Training script for WikiHop
#!/usr/bin/env bash

set -x
ORIG_DATA_DIR=$1
PREPROCESSED_DATA_DIR=$2
PATH_DIR=$3
PARAM_FILE=$4
FINAL_PATH_DUMPDIR="data/datasets/WikiHop/adjusted"
mkdir -p $FINAL_PATH_DUMPDIR

# Tokenization/tagging etc
scripts/preprocess_wikihop.sh $ORIG_DATA_DIR $PREPROCESSED_DATA_DIR

# break the train data
scripts/break_train_data_wikihop.py $PREPROCESSED_DATA_DIR ${PREPROCESSED_DATA_DIR}/train-split

# path extraction
scripts/path_finder_wikihop.sh $PREPROCESSED_DATA_DIR $PATH_DIR

# path adjustments
scripts/path_adjustments_wikihop.sh $PREPROCESSED_DATA_DIR $PATH_DIR $FINAL_PATH_DUMPDIR

# Preparing Vocabulary
allennlp make-vocab training_configs/config_wikihop_makevocab.json \
    -s data/datasets/WikiHop/ --include-package pathnet

# Training
MODELDIR="models/wikihop"
if [ -d $MODELDIR ]
then
    rm -r $MODELDIR
fi
mkdir -p $MODELDIR
allennlp train --file-friendly-logging -s $MODELDIR --include-package pathnet $PARAM_FILE
