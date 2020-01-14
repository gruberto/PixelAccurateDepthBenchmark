from __future__ import division
from Evaluator import Evaluator
from Dataset import Dataset
import os
import numpy as np
import multiprocessing
from metrics import Metric
#import tqdm
import sys

def print_result_visibility(result_visibility, delimiter='\t'):
    print(delimiter.join(['Visibility'] + Metric.get_header()))
    for i in range(result_visibility.shape[0]):
        print(delimiter.join(['{:10.3f}'.format(r) for r in list(result_visibility[i,:])]))

def print_result_rainfall(result_rainfall, delimiter='\t'):
    print(delimiter.join(['  Rainfall'] + Metric.get_header()))
    for i in range(result_rainfall.shape[0]):
        print(delimiter.join(['{:10.3f}'.format(r) for r in list(result_rainfall[i, :])]))

def print_result_clear(result, delimiter='\t'):
    print(delimiter.join(Metric.get_header()))
    for i in range(result.shape[0]):
        print(delimiter.join(['{:10.3f}'.format(r) for r in list(result[i, :])]))

def evaluate_mt(data_root, result_root, scenes, daytimes, approaches, evaluations, weathers, visibilities, rainfall_rates, nb_threads=8):

    d = Dataset(data_root)

    evaluate_jobs = []
    for weather in weathers:
        for evaluation in evaluations:
            for scene in scenes:
                for daytime in daytimes:
                    for approach in approaches:
                        if weather == 'fog':
                            for visibility in visibilities:
                                samples = d.get_fog_sequence(scene, daytime, visibility)
                                args = (data_root, result_root, samples, approach, evaluation)
                                evaluate_jobs.append(args)

                        if weather == 'rain':
                            for rainfall_rate in rainfall_rates:
                                samples = d.get_rain_sequence(scene, daytime, rainfall_rate)
                                args = (data_root, result_root, samples, approach, evaluation)
                                evaluate_jobs.append(args)

                        if weather == 'clear':
                            samples = d.get_clear_sequence(scene, daytime)
                            args = (data_root, result_root, samples, approach, evaluation)
                            evaluate_jobs.append(args)

    # pool = multiprocessing.Pool()  # this way all cores/threads of your machine are used
    print('{} evaluation jobs to run'.format(len(evaluate_jobs)))
    pool = multiprocessing.Pool(processes=nb_threads)  # specify number of processes if you are IO bound or have other reasons

    out = []
    for i, o in enumerate(pool.imap(evaluation_worker, evaluate_jobs, chunksize=1), 1):
        print('\rdone {0:%}'.format(i / len(evaluate_jobs)))
        out.append(o)

    counter = 0
    for weather in weathers:
        for evaluation in evaluations:
            print('##### {} #####'.format(evaluation))
            for scene in scenes:
                for daytime in daytimes:
                    for approach in approaches:
                        print('%%%%% {} %%%%%'.format(approach))
                        print(scene, daytime, weather)

                        if weather == 'fog':
                            result = np.array(out[counter:counter+len(visibilities)])
                            counter += len(visibilities)

                            if evaluation == 'binned_metrics':
                                result_visibility = result[:, 0, 1:]

                            elif evaluation == 'metrics':
                                result_visibility = result

                            result_visibility = np.hstack(
                                [np.array(visibilities).reshape((len(visibilities), 1)), result_visibility])

                            # save result
                            result_file = os.path.join(result_root, approach, '{}_{}_{}_{}_{}.txt'.format(approach, scene, daytime, weather, evaluation))
                            if not os.path.exists(os.path.split(result_file)[0]):
                                os.makedirs(os.path.split(result_file)[0])

                            header = 'Visibility,' + ','.join(Metric.get_header())
                            np.savetxt(result_file, result_visibility, delimiter=',', header=header)

                            print_result_visibility(result_visibility)

                        if weather == 'rain':
                            result = np.array(out[counter:counter + len(rainfall_rates)])
                            counter += len(rainfall_rates)

                            if evaluation == 'binned_metrics':
                                result_rainfall = result[:, 0, 1:]

                            elif evaluation == 'metrics':
                                result_rainfall = result

                            result_rainfall = np.hstack([np.array(rainfall_rates).reshape((len(rainfall_rates), 1)), result_rainfall])

                            # save result
                            result_file = os.path.join(result_root, approach, '{}_{}_{}_{}_{}.txt'.format(approach, scene, daytime, weather, evaluation))
                            if not os.path.exists(os.path.split(result_file)[0]):
                                os.makedirs(os.path.split(result_file)[0])

                            header = 'Rainfall,' + ','.join(Metric.get_header())
                            np.savetxt(result_file, result_rainfall, delimiter=',', header=header)

                            print_result_rainfall(result_rainfall)

                        if weather == 'clear':
                            result = np.array(out[counter])
                            counter += 1

                            if evaluation == 'metrics':
                                result_table = result.reshape((1, -1))

                            elif evaluation == 'binned_metrics':
                                result_table = result[0:1, 1:]

                                result_plot = result[1:, :]

                                result_plot_file = os.path.join(result_root, approach, '{}_{}_{}_{}_{}.txt'.format(approach, scene, daytime, weather, evaluation))
                                if not os.path.exists(os.path.split(result_plot_file)[0]):
                                    os.makedirs(os.path.split(result_plot_file)[0])

                                np.savetxt(result_plot_file, result_plot, delimiter=',')

                            # save result
                            result_file = os.path.join(result_root, approach, '{}_{}_{}_{}_{}.tex'.format(approach, scene, daytime, weather, evaluation))
                            if not os.path.exists(os.path.split(result_file)[0]):
                                os.makedirs(os.path.split(result_file)[0])

                            header = ','.join(Metric.get_header())
                            print_result_clear(result_table)

                            np.savetxt(result_file, result_table, delimiter=' & ', fmt='%.2f')


def evaluation_worker(args):
    data_root, result_root, samples, approach, evaluation = args
    e = Evaluator(data_root)

    if evaluation == 'metrics':
        metrics = e.evaluate_samples(samples, approach)
        return metrics

    if evaluation == 'binned_metrics':
        metrics = e.evaluate_samples_binned(samples, approach)
        return metrics



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate depth estimation results")
    parser.add_argument('--data_root', type=str, default='data', help='Path to data')
    parser.add_argument('--results_dir', type=str, default='results', help='Folder for evaluation results')
    parser.add_argument('--daytime', type=str, default='day', help='day or night')
    parser.add_argument('--approach', type=str, default='depth', help='Selected folder for evaluation')

    args = parser.parse_args()

    scenes = ['all'] #['all']
    evaluations = ['metrics', 'binned_metrics']
    weathers = ['clear', 'fog', 'rain']
    visibilities = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    rainfall_rates = [0, 15, 55]
    nb_threads = 8

    evaluate_mt(args.data_root, args.results_dir, scenes, [args.daytime], [args.approach], evaluations, weathers, visibilities, rainfall_rates, nb_threads=nb_threads)


