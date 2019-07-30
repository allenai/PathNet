#!/usr/bin/env bash

# path finder for splitted train files (WikiHop)
PREPROCESSED_DATA_DIR=$1
PATHDIR=$2
mkdir -p ${PATHDIR}/train-split

for x in $(seq 0 21); do
    echo "Split -- $x"
    echo "=============================="
    python scripts/prepro/path_finder_wikihop.py ${PREPROCESSED_DATA_DIR}/train-split/split_${x}.json ${PATHDIR}/train-split
done

# for dev
python scripts/prepro/path_finder_wikihop.py ${PREPROCESSED_DATA_DIR}/dev-processed-spacy.txt $PATHDIR