#!/bin/bash
src=$1
des=$2
name=$3

mkdir ~/Code/tx-muzero-hypermodel/results/${name}/2021${des}/
cp -a ~/save/${name}_*_2021${src}* ~/Code/tx-muzero-hypermodel/results/${name}/2021${des}/
rm -rf ~/save/${name}_*_2021${src}*

