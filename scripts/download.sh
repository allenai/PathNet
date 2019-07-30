#!/usr/bin/env bash

set -x

# Download GloVe embeddings
EMBDIR="data/embeddings"
mkdir -p $EMBDIR
wget -P $EMBDIR https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz

# Download WikiHop data
WH_DATA_DIR="data/datasets/WikiHop"
mkdir -p $WH_DATA_DIR
cd $WH_DATA_DIR
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA" -O qangaroo_v1.1.zip && rm -rf /tmp/cookies.txt
unzip qangaroo_v1.1.zip
mv qangaroo_v1.1/wikihop/train.json ./
mv qangaroo_v1.1/wikihop/dev.json ./  # alternaltively create a softlink
rm qangaroo_v1.1.zip
cd ../../..

# Download OBQA data
OBQA_DATA_DIR="data/datasets/OBQA"
mkdir -p $OBQA_DATA_DIR
wget -P $OBQA_DATA_DIR http://data.allenai.org/downloads/pathnet/obqa/inputs/obqa-commonsense-590k-wh-sorted100-train.json
wget -P $OBQA_DATA_DIR http://data.allenai.org/downloads/pathnet/obqa/inputs/obqa-commonsense-590k-wh-sorted100-dev.json
wget -P $OBQA_DATA_DIR http://data.allenai.org/downloads/pathnet/obqa/inputs/obqa-commonsense-590k-wh-sorted100-test.json

# Download the pretrained WikiHop Model
WH_MODEL_DIR="data/datasets/WikiHop/pretrained-model"
mkdir -p $WH_MODEL_DIR
wget -P $WH_MODEL_DIR http://data.allenai.org/downloads/pathnet/wikihop/model.tar.gz
