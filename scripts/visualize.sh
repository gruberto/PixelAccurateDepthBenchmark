#!/bin/bash

python src/visualize.py --approach lidar_hdl64_rgb_left --daytime day
python src/visualize.py --approach lidar_hdl64_rgb_left --daytime night

python src/visualize.py --approach psmnet --daytime day
python src/visualize.py --approach psmnet --daytime night

python src/visualize.py --approach sgm --daytime day
python src/visualize.py --approach sgm --daytime night

python src/visualize.py --approach sparse2dense --daytime day
python src/visualize.py --approach sparse2dense --daytime night

python src/visualize.py --approach monodepth --daytime day
python src/visualize.py --approach monodepth --daytime night
