import numpy as np
import matplotlib.pyplot as plt
import torch


def convert_equirect_to_camera_coord(depth_map, img_h, img_w):
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w),indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    # print("GGG ", depth_map.max(), depth_map.min())
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)


def get_3d_point_camera_coord(rotation_angle, listener_pos, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = listener_pos[0], listener_pos[1], listener_pos[2]
    # print(lis_x, lis_y, lis_z)
    if rotation_angle == 0:
        camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    elif rotation_angle == 90:
        camera_matrix = np.array([[0., 0., -1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([lis_z, -lis_y, -lis_x])
    elif rotation_angle == 180:
        camera_matrix = np.array([[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([lis_x, -lis_y, lis_z])
    elif rotation_angle == 270:
        camera_matrix = np.array([[0., 0., 1., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([-lis_z, -lis_y, lis_x])
    # point_3d[1] += 1.5528907
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]


def compute_energy_db(h):
    h = np.array(h)
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    return  energy_db

    