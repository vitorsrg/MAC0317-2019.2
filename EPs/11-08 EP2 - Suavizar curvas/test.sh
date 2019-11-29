#!/usr/bin/env bash

set -ev

if [ ! -d './.env' ]; then
  virtualenv ./.env -p $(which python3.7)
fi

source ./.env/bin/activate
export PATH=$(pwd)/.env/bin:$PATH

pip install scipy numpy imageio

tar -cvzf 10258862.tar.gz \
    ./samples ./fox_binary.png ./fox.png ./main.py \
    ./smoother2d.ipynb ./smoother2d.html ./README.md
