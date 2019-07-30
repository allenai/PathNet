#!/usr/bin/env bash

# path finder for splitted train files (OBQA)
PREPROCESSED_DATA_DIR=$1
PATHDIR=$2
DUMPDIR=$3

for y in 100; do
    echo "Numpaths -- $y"
    mkdir -p $DUMPDIR/paths${y}
    for x in train dev test; do
        echo "Split -- $x"
        echo "=============================="
        python scripts/prepro/obqa_prep_data_with_lemma.py $PREPROCESSED_DATA_DIR $PATHDIR $DUMPDIR/paths${y} \
            --mode $x --maxnumpaths $y
    done
done