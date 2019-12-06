#!/bin/bash

command -v unzip >/dev/null 2>&1 || { echo >&2 "'unzip' not installed. Install it via brew (i.e. 'brew insatll unzip' or 'sudo apt-get install unzip')"; exit 1; }
command -v kaggle >/dev/null 2>&1 || { echo >&2 "'kaggle' not installed. Switch to the right virual env and try running 'pip install kaggle'"; exit 1; }

CUR_DIR=$(pwd)
TARGET_DIR=$CUR_DIR/../data/

mkdir -p $TARGET_DIR
cd $TARGET_DIR

mkdir test_v2
mkdir train_v2

kaggle competitions download -c airbus-ship-detection

chmod 666 airbus-ship-detection.zip && unzip airbus-ship-detection.zip -d .

chmod 666 sample_submission_v2.csv
chmod 666 train_ship_segmentations_v2.csv

chmod 666 train_v2.zip && unzip train_v2.zip -d ./train_v2/. && rm -r train_v2.zip && \
chmod 666 test_v2.zip && unzip test_v2.zip -d ./test_v2/. && rm -r test_v2.zip && \
rm -r airbus-ship-detection.zip
