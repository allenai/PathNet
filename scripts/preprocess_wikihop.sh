#!/usr/bin/env bash

# Script to run preprocessing

export OPENBLAS_NUM_THREADS=1
set -e

DATA_DIR=$1
OUT_DIR=$2

python scripts/break_orig_wikihop_train.py $DATA_DIR/train.json

for x in $(seq 0 8); do
    python scripts/prepro/preprocess_wikihop.py $DATA_DIR $OUT_DIR \
        --split train${x} --num-workers 8
done

if [ -f "$OUT_DIR/train-processed-spacy.txt" ]
then
    rm $OUT_DIR/train-processed-spacy.txt
fi

for x in $(seq 0 8); do
    cat $OUT_DIR/train${x}-processed-spacy.txt >> $OUT_DIR/train-processed-spacy.txt
done

python scripts/prepro/preprocess_wikihop.py $DATA_DIR $OUT_DIR \
        --split dev --num-workers 8

#for x in train dev; do
#    echo "Split - $x"
#    python scripts/prepro/preprocess_wikihop.py $DATA_DIR $OUT_DIR \
#        --split $x --num-workers 8
#done

echo "Creating split directory"
mkdir -p ${OUT_DIR}/train-split
echo "Done"

# break the train data
python scripts/break_train_data_wikihop.py $OUT_DIR/train-processed-spacy.txt ${OUT_DIR}/train-split
