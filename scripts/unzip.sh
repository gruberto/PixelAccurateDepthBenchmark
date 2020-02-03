#!/bin/bash

download_root=$1
dst="data"

files=(
	$download_root/calibration.zip
	$download_root/gated0_10bit.zip
	$download_root/gated17_10bit.zip
	$download_root/gated31_10bit.zip
	$download_root/intermetric_gated.zip
	$download_root/intermetric_rgb_left.zip
	$download_root/lidar_hdl64_gated.zip
	$download_root/lidar_hdl64_rgb_left.zip
	$download_root/monodepth.zip
	$download_root/psmnet.zip
	$download_root/rgb_left_8bit.zip
	$download_root/rgb_right_8bit.zip
	$download_root/sgm.zip
	$download_root/sparse2dense.zip
)

mkdir -p $dst

all_exists=true
for item in ${files[*]} 
do
	if [[ ! -f "$item" ]]; then
    		echo "$item is missing"
		all_exists=false
	fi
done

if $all_exists; then
	for item in ${files[*]} 
	do
		unzip $item -d $dst/$filename
	done
fi
