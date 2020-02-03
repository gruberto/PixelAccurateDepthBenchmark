#!/bin/bash

python src/evaluate.py --approach lidar_hdl64_rgb_left --daytime day
python src/evaluate.py --approach lidar_hdl64_rgb_left --daytime night 

python src/evaluate.py --approach psmnet --daytime day
python src/evaluate.py --approach psmnet --daytime night

python src/evaluate.py --approach sgm --daytime day
python src/evaluate.py --approach sgm --daytime night

python src/evaluate.py --approach sparse2dense --daytime day
python src/evaluate.py --approach sparse2dense --daytime night

python src/evaluate.py --approach monodepth --daytime day
python src/evaluate.py --approach monodepth --daytime night
