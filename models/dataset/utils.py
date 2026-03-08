import numpy as np
import matplotlib.pyplot as plt
import torch


def generate_poisson_process(room_size = [5, 5, 5], imp_res_time = 0.5, fs = 22050, c=343):
    """generate Poisson random process"""
    #L: room length m
    # assume room volume (can be passed as a parameter)
    V = room_size[0] * room_size[1] * room_size[2]  # default room volume

    
    t0 = ((2 * V * np.log(2)) / (4 * np.pi * c**3))**(1/3)
    
    poisson_process = []
    time_values = []
    
    t = t0
    while t < imp_res_time:
        time_values.append(t)
        
        # determine polarity
        if (np.round(t * fs) - t * fs) < 0:
            poisson_process.append(1)
        else:
            poisson_process.append(-1)
        
        # determine average event rate
        mu = min(1e4, 4 * np.pi * c**3 * t**2 / V)
        
        # determine interval size
        delta_ta = (1 / mu) * np.log(1 / np.random.rand())
        t += delta_ta
    
    # create sampled random process
    rand_seq = np.zeros(int(imp_res_time * fs))
    for i, time_val in enumerate(time_values):
        idx = int(np.round(time_val * fs))
        if idx < len(rand_seq):
            rand_seq[idx] = poisson_process[i]
    rand_seq = torch.tensor(rand_seq).unsqueeze(0)
    #from IPython import embed; embed(using = False);os._exit(0)
    return rand_seq

def get_mean_room_xyz(depth):
    X = torch.max(depth[0,:,:]) - torch.min(depth[0,:,:])
    Y = torch.max(depth[1,:,:]) - torch.min(depth[1,:,:])
    Z = torch.max(depth[2,:,:]) - torch.min(depth[2,:,:])
    return (X.item(), Y.item(), Z.item())


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

if __name__ == "__main__":

    depth = torch.rand(3, 256, 256) * 4 
    room_size = get_mean_room_xyz(depth)
    #print(room_size)
    #visualize the poisson process
    imp_res_time = 12800/22050
    distance =3
    energy_dacay = 1/(distance)
    poisson_process = generate_poisson_process(room_size, imp_res_time, fs = 22050, c=343) * energy_dacay
    
    #print(poisson_process)
    rand_seq = poisson_process.squeeze(0).numpy()
   
    plt.figure(figsize=(12, 8))
    plt.xlim = [0, imp_res_time]
    plt.ylim = [-1.5, 1.5]
    plt.plot(rand_seq,linewidth=1)
    plt.savefig("../assets/poisson_process.pdf")
    