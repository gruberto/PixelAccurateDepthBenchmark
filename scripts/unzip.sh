#!/bin/bash

download_root=$1
dst="data"

mkdir $dst

for item in $download_root/*.zip
do
	filename=$(basename -- "$item")
	filename="${filename%.*}"
	mkdir -p $dst/$filename
	unzip $item -d $dst/$filename
done
