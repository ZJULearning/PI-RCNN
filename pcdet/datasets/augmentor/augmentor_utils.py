import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points, lidar_to_rect=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

        if lidar_to_rect is not None:
            lidar_to_rect = np.dot(lidar_to_rect, np.array([[1.,  0., 0., 0.],
                                                            [0., -1., 0., 0.],
                                                            [0.,  0., 1., 0.],
                                                            [0.,  0., 0., 1.]]))

    return gt_boxes, points, lidar_to_rect


def random_flip_along_y(gt_boxes, points, lidar_to_rect=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

        if lidar_to_rect is not None:
            lidar_to_rect = np.dot(lidar_to_rect, np.array([[-1., 0., 0., 0.],
                                                            [ 0., 1., 0., 0.],
                                                            [ 0., 0., 1., 0.],
                                                            [ 0., 0., 0., 1.]]))

    return gt_boxes, points, lidar_to_rect


def global_rotation(gt_boxes, points, rot_range, lidar_to_rect=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points, rot_matrix = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]), return_rot_matrix=True)
    points = points[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]
    if lidar_to_rect is not None:
        inv_rot_matrix = np.linalg.inv(rot_matrix).T
        # cosa = np.cos(-noise_rotation)
        # sina = np.sin(-noise_rotation)
        # inv_rot_matrix = np.array([
        #     [cosa, sina, 0.],
        #     [-sina, cosa, 0.],
        #     [0., 0., 1.]
        # ])
        padded_inv_rot_matrix = np.eye(4)
        padded_inv_rot_matrix[0:3, 0:3] = inv_rot_matrix
        lidar_to_rect = np.dot(lidar_to_rect, padded_inv_rot_matrix)
    return gt_boxes, points, lidar_to_rect


def global_scaling(gt_boxes, points, scale_range, lidar_to_rect=None):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    if lidar_to_rect is not None:
        a = 1 / noise_scale
        lidar_to_rect = np.dot(lidar_to_rect, np.array([[ a, 0., 0., 0.],
                                                        [0.,  a, 0., 0.],
                                                        [0., 0.,  a, 0.],
                                                        [0., 0., 0., 1.]]))
    return gt_boxes, points, lidar_to_rect
