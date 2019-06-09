#!/bin/bash

if [ ! -d "${HOME}/data" ]; then
  mkdir ${HOME}/data
fi

read  -r -p "Downlaod ISIC2019 Data - press key to continue" key
if [ ! -d "${HOME}/data/ISIC2019" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  read  -r -p "ISIC2019 Data not detected on your ${HOME}/data/ISIC2019 - press key to download" key
  # https://drive.google.com/file/d/1saXT05R3xMTDwCnv-M7Wz5o0h8usyKz4/view?usp=sharing
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1saXT05R3xMTDwCnv-M7Wz5o0h8usyKz4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1saXT05R3xMTDwCnv-M7Wz5o0h8usyKz4" -O isic2019_train.tar.gz && rm -rf /tmp/cookies.txt
  mv isic2019_train.tar.gz ${HOME}/data
  tar -zxvf ${HOME}/data/isic2019_train.tar.gz -C ${HOME}/data
  rm ${HOME}/data/isic2019_train.tar.gz
else
  # Control will enter here if $DIRECTORY exists.
  echo "ISIC2019 Data Detected on your ${HOME}/data/ISIC2019!"
fi