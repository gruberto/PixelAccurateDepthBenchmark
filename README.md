DepthBenchmark
============================
This repo contains the code and data for [Pixel-Accurate Depth Evaluation in Realistic Driving Scenarios](https://arxiv.org/pdf/1906.08953.pdf) by [Tobias Gruber](https://scholar.google.de/citations?user=w-VeQ5cAAAAJ&hl=de), [Mario Bijelic](http://mariobijelic.de/wordpress/), [Felix Heide](http://www.cs.princeton.edu/~fheide/), [Werner Ritter](https://www.xing.com/profile/Werner_Ritter7) and [Klaus Dietmayer](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/institute/staff/institutional-administration/prof-dr-ing-klaus-dietmayer/).

Code and data will be available soon.

## Abstract
This work presents an evaluation benchmark for depth estimation and completion using high-resolution depth measurements with angular resolution of up to 25" (arcsecond), akin to a 50 megapixel camera with per-pixel depth available. Existing datasets, such as the KITTI benchmark, provide only sparse reference measurements with an order of magnitude lower angular resolution - these sparse measurements are treated as ground truth by existing depth estimation methods. We propose an evaluation in four characteristic automotive scenarios recorded in varying weather conditions (day, night, fog, rain). As a result, our benchmark allows to evaluate the robustness of depth sensing methods to adverse weather and different driving conditions. Using the proposed evaluation data, we show that current stereo approaches provide significantly more stable depth estimates than monocular methods and lidar completion in adverse weather.

## Reference
If you find our work on gated depth estimation useful in your research, please consider citing our paper:
```
@inproceedings{depthbenchmark2019,
  title     = {Pixel-Accurate Depth Evaluation in Realistic Driving Scenarios},
  author    = {Gruber, Tobias and Bijelic, Mario and Heide, Felix and Ritter, Werner and Dietmayer, Klaus},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2019}
}
```
