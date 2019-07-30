# script for answer prediction on WikiHop dev/test data
#!/usr/bin/env bash

set -x
set -e

DATA_DIR=$1
MODEL_FILE=$2
OUTFILE=$3
EMBFILE="data/embeddings/glove.840B.300d.txt.gz"

PREPROCESSED_DATA_DIR="data/datasets/WikiHop/prepro-data"
PATHDIR="data/datasets/WikiHop/dev-paths"
DUMPDIR="data/datasets/WikiHop/dev-adjusted-paths"
PREDDIR="data/datasets/WikiHop/model-preds"
UPDATED_MODEL_DIR="data/datasets/WikiHop/models-updated"


echo "Creating Data Directory"
mkdir -p $PREPROCESSED_DATA_DIR
echo "Preprocessing Data"
python scripts/prepro/preprocess_wikihop.py $DATA_DIR $PREPROCESSED_DATA_DIR --split dev \
    --num-workers 6

echo "Path Finding"
mkdir -p $PATHDIR
python scripts/prepro/path_finder_wikihop.py $PREPROCESSED_DATA_DIR/dev-processed-spacy.txt $PATHDIR

echo "Path Adjusting"
mkdir -p $DUMPDIR
python scripts/prepro/wikihop_prep_data_with_lemma.py $PREPROCESSED_DATA_DIR $PATHDIR $DUMPDIR \
        --mode dev --maxnumpaths 30

echo "Expanding vocabulary"
mkdir -p $UPDATED_MODEL_DIR
python scripts/expand_vocabulary.py --file $DUMPDIR/dev-path-lines.txt \
    --emb_wt_key _text_field_embedder.token_embedder_tokens.weight \
    --embeddings $EMBFILE \
    --model $MODEL_FILE \
    --output_dir $UPDATED_MODEL_DIR

echo "Running Allennlp Prediction"
mkdir -p $PREDDIR
CUDA_VISIBLE_DEVICES=None allennlp predict --output-file $PREDDIR/dev_predictions.txt \
    --predictor wikihop_predictor \
    --batch-size 3 --include-package pathnet --cuda-device -1 \
    --silent $UPDATED_MODEL_DIR/model.tar.gz \
    $DUMPDIR/dev-path-lines.txt

echo "Preparing Prediction Dictionary"
python scripts/prepare_outfile.py $PREDDIR/dev_predictions.txt $OUTFILE
