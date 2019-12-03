import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import cv2


def colorize_pointcloud(depth, min_distance=3, max_distance=80, radius=3):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(depth > 0)

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)

    return pointcloud_color


def colorize_pointcloud_emphasize_clutter(depth, min_distance=3, max_distance=80, radius=3, threshold=15):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(depth > 0)

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        if depth[pos[i, 0], pos[i, 1]] < threshold and pos[i, 0] < 550:
            r = 2 * radius
        else:
            r = radius
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), r, (color[0], color[1], color[2]), -1)

    return pointcloud_color

def colorize_error(depth, groundtruth, min_distance=3, max_distance=80, radius=3, threshold=5):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    pos = np.argwhere(np.logical_and(groundtruth > 0, depth > 0))
    error = np.abs(depth[pos[:,0], pos[:,1]] - groundtruth[pos[:,0], pos[:,1]])

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        # color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        color = (255, 255, 255)
        if error[i] > threshold:
            cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)

    return pointcloud_color


def colorize_depth(depth, min_distance=3, max_distance=80):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
    depth_color[depth <= 0] = [0, 0, 0]
    depth_color[np.isnan(depth)] = [0, 0, 0]
    depth_color[depth == np.inf] = [0, 0, 0]

    return depth_color


def crop_rgb_to_gated(rgb, gated_width=1280, gated_height=720):
    rgb = rgb[215:980, 230:1590]
    rgb = cv2.resize(rgb, (gated_width, gated_height), interpolation=cv2.INTER_AREA)

    return rgb
