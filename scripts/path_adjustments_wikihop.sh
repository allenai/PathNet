#!/usr/bin/env bash

# path finder for splitted train files (WikiHop)
PREPROCESSED_DATA_DIR=$1
PATHDIR=$2
DUMPDIR=$3

for y in 30; do
    mkdir -p $DUMPDIR/paths${y}
    for x in train dev; do
        echo "Split -- $x"
        echo "Numpaths -- $y"
        echo "=============================="
        python scripts/prepro/wikihop_prep_data_with_lemma.py $PREPROCESSED_DATA_DIR $PATHDIR $DUMPDIR/paths${y} \
            --mode $x --maxnumpaths $y
    done
done
