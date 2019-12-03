import numpy as np
from skimage.measure import compare_ssim
import math

class Metric:

    def __init__(self):
        self.min_val = 1e-7

    def threshold(self, y1, y2, thr=1.25):
        max_ratio = np.maximum(y1 / y2, y2 / y1)
        return np.mean(max_ratio < thr, dtype=np.float64) * 100.

    def rmse(self, y1, y2):
        diff = y1 - y2
        return math.sqrt(np.mean(diff * diff, dtype=np.float64))

    def trmse(self, y1, y2, thr=5):
        diff = y1 - y2
        return math.sqrt(np.mean(np.minimum(diff * diff, thr * thr), dtype=np.float64))

    def rmse_log(self, y1, y2):
        return self.rmse(np.log(y1), np.log(y2))

    def ard(self, y1, y2):
        return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64) * 100

    def srd(self, y1, y2):
        return np.mean((y1 - y2) ** 2 / y2, dtype=np.float64)

    def mae(self, y1, y2):
        return np.mean(np.abs(y1 - y2), dtype=np.float64)

    def tmae(self, y1, y2, thr=5):
        return np.mean(np.minimum(np.abs(y1 - y2), thr), dtype=np.float64)

    def ssim(self, y1, y2):
        return compare_ssim(y1, y2)

    def psnr(self, y1, y2):
        rmse = self.rmse(y1, y2)
        if rmse == 0:
            return 100
        return 20 * math.log10(np.amax( np.abs(y1 - y2)) / rmse)

    def rpsnr(self, y1, y2):
        srd = self.srd(y1, y2)
        if srd == 0:
            return 100
        return 20 * math.log10(np.amax( np.abs(y1 - y2) / y2) / math.sqrt(srd))

    def silog(self, y1, y2):
        d = np.log(y1) - np.log(y2)
        return 100 * math.sqrt(np.mean(d ** 2, dtype=np.float64) - (np.mean(d, dtype=np.float64)) ** 2)

    def get_eval_pos(self, output, groundtruth):
        return np.logical_and(groundtruth > 0, output > 0)

    def clip(self, output, groundtruth, clip_min, clip_max):
        output_clipped = np.clip(output, clip_min, clip_max)
        groundtruth_clipped = np.clip(groundtruth, clip_min, clip_max)
        return output_clipped, groundtruth_clipped

    def calc_metrics(self, output, groundtruth, clip_min=0., clip_max=25.):

        eval_pos = np.logical_and(groundtruth > 0, output > 0)
        ground_truth_points = np.sum(groundtruth > 0)
        output = output[eval_pos]
        groundtruth = groundtruth[eval_pos]

        output, groundtruth = self.clip(output, groundtruth, clip_min=clip_min, clip_max=clip_max)

        return self.rmse(output, groundtruth), \
               self.trmse(output, groundtruth), \
               self.mae(output, groundtruth), \
               self.tmae(output, groundtruth), \
               self.rmse_log(output, groundtruth), \
               self.srd(output, groundtruth), \
               self.ard(output, groundtruth), \
               self.silog(output, groundtruth),  \
               self.threshold(output, groundtruth, thr=1.25), \
               self.threshold(output, groundtruth, thr=1.25 ** 2), \
               self.threshold(output, groundtruth, thr=1.25 ** 3), \
               self.ssim(output, groundtruth), \
               self.psnr(output, groundtruth), \
               self.rpsnr(output, groundtruth), \

    @staticmethod
    def get_header():
        metrics = ['RMSE', 'tRMSE', 'MAE', 'tMAE', 'RMSElog', 'SRD', 'ARD', 'SIlog', 'delta1', 'delta2', 'delta3', 'SSIM', 'PSNR', 'rPSNR']
        maxlen = 10
        metrics = [(' ' * (maxlen - len(x))) + x for x in metrics]
        return metrics


if __name__ == '__main__':
    y = np.array(range(10, 130)).reshape((10, 12))
    noise = np.ones_like(y) * 10
    noise[np.random.random(size=noise.shape) > 0.5] *= -2
    y_noisy = y + noise
    print(y)
    print(y_noisy)
    m = Metric()
    print(m.calc_metrics(y_noisy, y))
	

