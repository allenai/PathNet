#!/usr/bin/env bash
# Script to run preprocessing

export OPENBLAS_NUM_THREADS=1
set -e

ORIG_DATA_DIR=$1
PREPROCESSED_DATA_DIR=$2

for x in train dev test; do
    echo "Split - $x"
    # Change the filename accoridngly
    python scripts/prepro/preprocess_obqa.py $ORIG_DATA_DIR/obqa-commonsense-590k-wh-sorted100-${x} \
        $PREPROCESSED_DATA_DIR \
        --split $x --num-workers 6
done

echo "Creating split directory"
mkdir -p ${PREPROCESSED_DATA_DIR}/train-split
echo "Done"

# break the train data
python scripts/break_train_data_obqa.py $PREPROCESSED_DATA_DIR/train-processed-spacy.txt \
    ${PREPROCESSED_DATA_DIR}/train-split