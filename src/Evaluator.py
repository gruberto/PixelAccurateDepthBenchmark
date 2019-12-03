from metrics import Metric
import os
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import json
from scipy.stats import binned_statistic_2d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import time

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Evaluator:
    def __init__(self, data_root, rgb_shape=(1024, 1920), gated_shape=(720, 1280), bfl_rgb=0.202993 * 2355.722801,
                 bfl_gated=0.202993 * 2322.4, clip_min=0.001, clip_max=28., nb_bins=14):

        self.data_root = data_root
        self.gated_height, self.gated_width = gated_shape
        self.rgb_height, self.rgb_width = rgb_shape
        self.bfl_rgb = bfl_rgb  # baseline times focal length
        self.bfl_gated = bfl_gated  # baseline times focal length
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.metric = Metric()

        self.fov_h_camera = 40
        self.fov_v_camera = 23

        self.nb_metrics = len(Metric.get_header())

        self.bins, self.mean_bins = self.calc_bins(self.clip_min, self.clip_max, nb_bins)

        self.frames = {
            'monodepth': 'rgb_left',
            'semodepth': 'rgb_left',
            'psmnet': 'rgb_left',
            'unsup': 'rgb_left',
            'sparse2dense': 'rgb_left',
            'sgm': 'rgb_left',
            'lidar_hdl64_rgb_left': 'rgb_left',
            'depth': 'rgb_left',
            'intermetric': 'rgb_left',
        }

        self.crops = {
            # 'rgb_left': [215, 330, 44, 230],  # top, right, bottom, left
            'rgb_left': [270, 20, 20, 170],  # top, right, bottom, left
            'gated': [0, 0, 0, 0],  # top, right, bottom, left
        }

    @staticmethod
    def calc_bins(clip_min, clip_max, nb_bins):
        bins = np.linspace(clip_min, clip_max, num=nb_bins + 1)
        mean_bins = np.array([0.5 * (bins[i + 1] + bins[i]) for i in range(0, nb_bins)])
        return bins, mean_bins

    @staticmethod
    def interpolate(depth, method='nearest'):
        current_coordinates = np.where(depth > 0)
        missing_coordinates = np.where(depth == 0)

        if len(missing_coordinates[0]) > 0:
            current_depth = depth[current_coordinates]
            if method == 'nearest':
                interpolator = NearestNDInterpolator(current_coordinates, current_depth)
            elif method == 'linear':
                interpolator = LinearNDInterpolator(current_coordinates, current_depth, )
            depth[missing_coordinates] = interpolator(missing_coordinates)

        return depth

    def disparity2depth(self, disparity, domain='rgb'):
        depth = np.zeros(disparity.shape)
        depth[disparity == 0] = float('inf')
        if domain == 'rgb':
            depth[disparity != 0] = self.bfl_rgb / disparity[disparity != 0]
        if domain == 'gated':
            depth[disparity != 0] = self.bfl_gated / disparity[disparity != 0]

        return depth

    def crop_rgb_to_gated(self, rgb):
        rgb = rgb[215:980, 230:1590]
        return rgb

    def crop_eval_region(self, img, frame):
        crop = self.crops[frame]
        img = img[crop[0]:(img.shape[0] - crop[2]), crop[3]:(img.shape[1] - crop[1])]

        return img

    def evaluate_samples(self, samples, approach):
        metrics = np.zeros((len(samples), self.nb_metrics))
        for idx, sample in enumerate(samples):
            metrics[idx, :] = self.evaluate_sample(sample, approach, gt_type='intermetric')
        metrics = np.mean(metrics, axis=0)

        return metrics

    def evaluate_samples_binned(self, samples, approach):
        metrics = np.zeros((len(samples), len(self.mean_bins) + 1, self.nb_metrics + 1))
        for idx, sample in enumerate(samples):
            metrics[idx, :, :] = self.evaluate_sample_binned(sample, approach, gt_type='intermetric')
        metrics = np.mean(metrics, axis=0)

        return metrics

    def evaluate_sample(self, sample, approach, gt_type='intermetric'):
        depth_gt = self.load_depth_groundtruth(sample.split('_')[0], frame=self.frames[approach], gt_type=gt_type)
        depth = self.load_depth(sample, approach)
        return self.metric.calc_metrics(depth, depth_gt, clip_min=self.clip_min, clip_max=self.clip_max)

    def evaluate_sample_binned(self, sample, approach, gt_type='intermetric'):
        depth_gt = self.load_depth_groundtruth(sample.split('_')[0], frame=self.frames[approach],
                                               gt_type=gt_type).flatten()
        depth = self.load_depth(sample, approach).flatten()

        results = np.vstack([depth, depth_gt])

        if len(self.bins) == 1:
            inds = np.zeros((results.shape[-1],), dtype=int)
        else:
            inds = np.digitize(results[1, :], self.bins)

        error_binned = np.zeros((len(self.bins) - 1, self.nb_metrics))
        for i, bin in enumerate(self.bins[:-1]):
            try:
                metrics = self.metric.calc_metrics(results[0, inds == i], results[1, inds == i], clip_min=self.clip_min,
                                                   clip_max=self.clip_max)
                error_binned[i, :] = metrics
            except ValueError:
                error_binned[i, :] = np.nan

        mean_error_binned = np.zeros((self.nb_metrics,))
        for i in range(0, self.nb_metrics):
            mean_error_binned[i] = np.mean(error_binned[~np.isnan(error_binned[:, i]), i])

        error_binned = np.hstack([self.mean_bins.reshape((-1, 1)), error_binned])
        mean_error_binned = np.hstack([np.zeros((1,)), mean_error_binned])
        metric = np.vstack([mean_error_binned, error_binned])

        return metric

    def error_image(self, sample, approach, gt_type='intermetric'):
        groundtruth = self.load_depth_groundtruth(sample.split('_')[0], frame=self.frames[approach], gt_type=gt_type)
        output = self.load_depth(sample, approach)

        error_image = np.zeros(groundtruth.shape)

        eval_pos = np.logical_and(groundtruth > 0, output > 0)
        output = output[eval_pos]
        groundtruth = groundtruth[eval_pos]
        output = np.clip(output, self.clip_min, self.clip_max)
        groundtruth = np.clip(groundtruth, self.clip_min, self.clip_max)

        error_image[eval_pos] = np.abs(output - groundtruth)

        norm = mpl.colors.Normalize(vmin=0, vmax=self.clip_max - self.clip_min)
        cmap = cm.hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        error_image = m.to_rgba(error_image)[:, :, 0:3]
        error_image[~eval_pos] = (0, 0, 0)

        error_image = cv2.cvtColor((255 * error_image).astype(np.uint8), cv2.COLOR_BGR2RGB)

        return error_image

    def depth2points(self, depth, P):
        inv_projection_matrix = np.linalg.inv(P)
        idx = np.indices((depth.shape[0], depth.shape[1]))
        index_matrix = np.vstack((idx[1].flatten(), idx[0].flatten(), np.ones((depth.shape[0] * depth.shape[1]))))
        points = depth.flatten() * np.matmul(inv_projection_matrix, index_matrix)
        points = points[:, depth.flatten() != 0]

        return points

    def create_top_view(self, sample, approach):

        output = self.load_depth(sample, approach)

        if self.frames[approach] == 'rgb_left':
            calib_file = 'calib/calib_cam_stereo_left.json'
        elif self.frames[approach] == 'gated':
            calib_file = 'calib/calib_gated_bwv.json'

        with open(calib_file, 'r') as f:
            calib = json.load(f)

        P = np.array(calib['P']).reshape(3, 4)[:, 0:3]
        P_scaling = np.array([[1, 0, -self.crops[self.frames[approach]][3]], [0, 1, -self.crops[self.frames[approach]][0]], [0, 0, 1]])
        P = np.dot(P_scaling, P)

        pc = self.depth2points(output, P).transpose()
        temp = pc.copy()

        pc[:, 0] = temp[:, 2]  # x to the front
        pc[:, 1] = -temp[:, 0]  # y to the left
        pc[:, 2] = -temp[:, 1]  # z to the top

        fov_h_camera = 40
        fov_v_camera = 23

        xbins = 231
        ybins = 101
        bins = [np.linspace(5, 28, xbins), np.linspace(-5, 5, ybins)]

        top_view, xedges, yedges, _ = binned_statistic_2d(pc[:, 0], pc[:, 1], pc[:, 2], 'count', bins)

        xbins_middle = np.array([0.5 * (xedges[i] + xedges[i + 1]) for i in range(len(xedges) - 1)])

        fov_area = np.tan(0.5 * np.deg2rad(fov_h_camera)) * np.tan(
            0.5 * np.deg2rad(self.fov_v_camera)) * xbins_middle ** 2
        fov_area = np.tile(fov_area, (ybins - 1, 1)).transpose()

        top_view = fov_area * top_view

        top_view_norm = mpl.colors.Normalize(vmin=0, vmax=pc.shape[0] / 220.0)
        t = cm.ScalarMappable(norm=top_view_norm, cmap=cm.jet)

        top_view_color = t.to_rgba(top_view)[:, :, 0:3]

        top_view = np.rot90(top_view, axes=(0, 1))
        top_view_color = np.rot90(top_view_color, axes=(0, 1))

        top_view_color = cv2.cvtColor((255 * top_view_color).astype(np.uint8), cv2.COLOR_BGR2RGB)

        return top_view, top_view_color

    def load_depth_groundtruth(self, scene, frame='rgb_left', gt_type='intermetric'):
        if gt_type == 'intermetric':
            path = os.path.join(self.data_root, 'intermetric_{}'.format(frame), scene + '.npz')
        elif gt_type == 'lidar':
            path = os.path.join(self.data_root, 'lidar_hdl64_{}'.format(frame), scene + '.npz')

        depth = np.load(path)['arr_0']

        depth = self.crop_eval_region(depth, frame)

        return depth

    def load_depth(self, sample, approach, interpolate=True):

        if approach == 'intermetric':
            path = os.path.join(self.data_root, approach + '_' + self.frames[approach], sample + '.npz')
        else:
            path = os.path.join(self.data_root, approach, sample + '.npz')
        depth = np.load(path)['arr_0']
        depth = self.pp_depth(depth, approach, interpolate=interpolate)

        return depth

    def pp_depth(self, depth, approach, interpolate=True):

        if approach == 'monodepth':
            disparity = depth
            disparity = cv2.resize(disparity, (self.rgb_width, self.rgb_height), interpolation=cv2.INTER_AREA)
            disparity = disparity * self.rgb_width  # it provides relative disparity
            depth = self.disparity2depth(disparity, domain='rgb')

        elif approach == 'semodepth':
            disparity = np.squeeze(depth)
            # disparity = np.squeeze(disparity) / 960
            disparity = cv2.resize(disparity, (self.rgb_width, self.rgb_height), interpolation=cv2.INTER_AREA)
            # disparity = disparity * self.rgb_width
            depth = self.disparity2depth(disparity, domain='rgb')

        elif approach == 'psmnet':
            disparity = depth
            disparity = disparity / 960
            disparity = cv2.resize(disparity, (self.rgb_width, self.rgb_height), interpolation=cv2.INTER_AREA)
            disparity = disparity * self.rgb_width
            depth = self.disparity2depth(disparity, domain='rgb')

        elif approach == 'unsup':
            disparity = depth
            disparity = cv2.resize(disparity, (self.rgb_width, self.rgb_height), interpolation=cv2.INTER_AREA)
            disparity = disparity * self.rgb_width  # it provides relative disparity
            depth = self.disparity2depth(disparity, domain='rgb')

        elif approach == 'gated2depth':
            depth = np.squeeze(depth['arr_0'])

        elif approach == 'sgm':
            disparity = depth
            disparity_full = np.zeros((1024, 1920))
            disparity_full[70:70 + 824, :] = cv2.resize(disparity, (1920, 824))
            depth = self.disparity2depth(2 * disparity_full, 'rgb')

        depth[np.isnan(depth)] = 0.0
        depth[depth == np.inf] = 0.0

        if interpolate:
            depth = self.interpolate(depth)

        depth = self.crop_eval_region(depth, self.frames[approach])
        depth = depth.astype(np.float32)

        return depth

    def load_rgb(self, sample):
        path = os.path.join(self.data_root, 'rgb_left_8bit', sample + '.png')
        rgb = cv2.imread(path)
        rgb = self.crop_eval_region(rgb, 'rgb_left')

        return rgb

    def load_gated(self, sample, slice):
        path = os.path.join(self.data_root, 'gated{}_8bit'.format(slice), sample + '.png')
        img = cv2.imread(path)
        img = self.crop_eval_region(img, 'gated')

        return img
