from Evaluator import Evaluator
from results import colorize_pointcloud, colorize_depth
from Dataset import Dataset
import cv2
import os
import numpy as np

def visualize(data_root, result_root, scenes, daytimes, approaches, evaluations, weathers, visibilities, rainfall_rates):

    d = Dataset(data_root)
    e = Evaluator(data_root)

    for scene in scenes:
        if 'intermetric' in evaluations:

            depth = e.load_depth_groundtruth(scene, frame='rgb_left', gt_type='intermetric')
            depth_color = colorize_depth(depth, min_distance=e.clip_min, max_distance=e.clip_max)

            intermetric_path = os.path.join(result_root, 'intermetric', '{}_{}.jpg'.format('intermetric', scene))
            if not os.path.exists(os.path.split(intermetric_path)[0]):
                os.makedirs(os.path.split(intermetric_path)[0])
            cv2.imwrite(intermetric_path, depth_color)

            # top_view, top_view_color = e.create_top_view(e.load_depth_groundtruth, scene)
            top_view, top_view_color = e.create_top_view(scene, 'intermetric')
            intermetric_top_view_file = os.path.join(result_root, 'intermetric',
                                                     '{}_{}_topview.jpg'.format('intermetric', scene))
            cv2.imwrite(intermetric_top_view_file, top_view_color)

        for daytime in daytimes:
            for weather in weathers:
                samples = []

                if weather == 'fog':
                    for visibility in visibilities:
                        samples.append(d.get_fog_sequence(scene, daytime, visibility)[0])

                if weather == 'rain':
                    for rainfall_rate in rainfall_rates:
                        samples.append(d.get_rain_sequence(scene, daytime, rainfall_rate)[0])

                if weather == 'clear':
                    samples.append(d.get_clear_sequence(scene, daytime)[0])

                for i, sample in enumerate(samples):
                    print(sample)

                    if 'rgb' in evaluations:
                        rgb = e.load_rgb(sample)

                        if weather == 'fog':
                            rgb_path = os.path.join(result_root, 'rgb',
                                                    '{}_{}_{}_{}_{}'.format('rgb', scene, daytime, weather,
                                                                            visibilities[i]))
                        elif weather == 'rain':
                            rgb_path = os.path.join(result_root, 'rgb',
                                                    '{}_{}_{}_{}_{}'.format('rgb', scene, daytime, weather,
                                                                            rainfall_rates[i]))
                        elif weather == 'clear':
                            rgb_path = os.path.join(result_root, 'rgb',
                                                    '{}_{}_{}_{}'.format('rgb', scene, daytime, weather))

                        if not os.path.exists(os.path.split(rgb_path)[0]):
                            os.makedirs(os.path.split(rgb_path)[0])

                        cv2.imwrite(rgb_path + '.jpg', rgb)

                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        rgb[:, :, 0] = clahe.apply(rgb[:, :, 0])
                        rgb[:, :, 1] = clahe.apply(rgb[:, :, 1])
                        rgb[:, :, 2] = clahe.apply(rgb[:, :, 2])

                        cv2.imwrite(rgb_path + '_clahe.jpg', rgb)

                    if 'lidar_raw' in evaluations:

                        depth = e.load_depth(sample, 'lidar_hdl64_rgb_left', interpolate=False)
                        depth_color = colorize_pointcloud(depth, min_distance=e.clip_min, max_distance=e.clip_max,
                                                          radius=5)

                        if weather == 'fog':
                            lidar_path = os.path.join(result_root, 'lidar_raw',
                                                      '{}_{}_{}_{}_{}'.format('lidar_raw', scene, daytime, weather,
                                                                              visibilities[i]))
                        elif weather == 'rain':
                            lidar_path = os.path.join(result_root, 'lidar_raw',
                                                      '{}_{}_{}_{}_{}'.format('lidar_raw', scene, daytime, weather,
                                                                              rainfall_rates[i]))
                        elif weather == 'clear':
                            lidar_path = os.path.join(result_root, 'lidar_raw',
                                                      '{}_{}_{}_{}'.format('lidar_raw', scene, daytime, weather))

                        if not os.path.exists(os.path.split(lidar_path)[0]):
                            os.makedirs(os.path.split(lidar_path)[0])

                        cv2.imwrite(lidar_path + '.jpg', depth_color)

                    if 'gated' in evaluations:
                        for t in [0,17,31]:

                            gated_img = e.load_gated(sample, t)

                            if weather == 'fog':
                                gated_path = os.path.join(result_root, 'gated{}'.format(t),
                                                          '{}_{}_{}_{}_{}'.format('gated{}'.format(t), scene, daytime, weather,
                                                                                  visibilities[i]))
                            elif weather == 'rain':
                                gated_path = os.path.join(result_root, 'gated{}'.format(t),
                                                          '{}_{}_{}_{}_{}'.format('gated{}'.format(t), scene, daytime, weather,
                                                                                  rainfall_rates[i]))
                            elif weather == 'clear':
                                gated_path = os.path.join(result_root, 'gated{}'.format(t),
                                                          '{}_{}_{}_{}'.format('gated{}'.format(t), scene, daytime, weather))

                            if not os.path.exists(os.path.split(gated_path)[0]):
                                os.makedirs(os.path.split(gated_path)[0])

                            cv2.imwrite(gated_path + '.jpg', gated_img)

                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            gated_img[:, :, 0] = clahe.apply(gated_img[:, :, 0])
                            gated_img[:, :, 1] = clahe.apply(gated_img[:, :, 1])
                            gated_img[:, :, 2] = clahe.apply(gated_img[:, :, 2])

                            cv2.imwrite(gated_path + '_clahe.jpg', gated_img)

                    for approach in approaches:
                        if weather == 'fog':
                            sample_path = os.path.join(result_root, approach,
                                                       '{}_{}_{}_{}_{}'.format(approach, scene, daytime, weather,
                                                                               visibilities[i]))
                        elif weather == 'rain':
                            sample_path = os.path.join(result_root, approach,
                                                       '{}_{}_{}_{}_{}'.format(approach, scene, daytime, weather,
                                                                               rainfall_rates[i]))
                        elif weather == 'clear':
                            sample_path = os.path.join(result_root, approach,
                                                       '{}_{}_{}_{}'.format(approach, scene, daytime, weather))

                        if not os.path.exists(os.path.split(sample_path)[0]):
                            os.makedirs(os.path.split(sample_path)[0])

                        if 'depth_map' in evaluations:
                            depth = e.load_depth(sample, approach)

                            depth_color = colorize_depth(depth, min_distance=e.clip_min, max_distance=e.clip_max)
                            depth_map_path = sample_path + '_depth_map.jpg'
                            cv2.imwrite(depth_map_path, depth_color)

                        if 'error_image' in evaluations:
                            error_image = e.error_image(sample, approach, gt_type='intermetric')
                            error_image_path = sample_path + '_error_image.jpg'
                            cv2.imwrite(error_image_path, error_image)

                        if 'top_view' in evaluations:
                            top_view, top_view_color = e.create_top_view(sample, approach)
                            top_view_path = sample_path + '_top_view.jpg'
                            cv2.imwrite(top_view_path, top_view_color)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Visualize depth estimation results")
    parser.add_argument('--data_root', type=str, default='data', help='Path to data')
    parser.add_argument('--results_dir', type=str, default='results', help='Folder for evaluation results')
    parser.add_argument('--daytime', type=str, default='day', help='day or night')
    parser.add_argument('--approach', type=str, default='depth', help='Selected folder for evaluation')

    args = parser.parse_args()

    scenes = ['scene1', 'scene2', 'scene3', 'scene4']
    daytimes = ['day', 'night']
    evaluations = ['depth_map', 'rgb', 'lidar_raw', 'intermetric', 'top_view']
    weathers = ['clear', 'fog', 'rain']
    visibilities = [20, 40, 30, 50, 70, 100]
    rainfall_rates = [0, 15, 55]

    visualize(args.data_root, args.results_dir, scenes, daytimes, [args.approach], evaluations, weathers, visibilities, rainfall_rates)


