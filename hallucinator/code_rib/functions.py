import torch
import numpy as np
import torch.nn as nn
from quaternion import qeuler_np
from remove_fs import remove_fs

def PLU(x, alpha = 0.1, c = 1.0):
    relu = nn.ReLU()
    o1 = alpha * (x + c) - c
    o2 = alpha * (x - c) + c
    o3 = x - relu(x - o2)
    o4 = relu(o1 - o3) + o3
    return o4

def gen_ztta(dim = 256, length = 50):
    ### currently without T_max ###
    ztta = np.zeros((1, length, dim))
    for t in range(length):
        for d in range(dim):
            if d % 2 == 0:
                ztta[:, t, d] = np.sin(1.0 * (length - t) / 10000 ** (d / dim))
            else:
                ztta[:, t, d] = np.cos(1.0 * (length - t) / 10000 ** (d / dim))
    return torch.from_numpy(ztta.astype(np.float32))

def gen_ztar(sigma = 1.0, length = 50):
    ### currently noise term in not inroduced ###
    lambda_tar = []
    for t in range(length):
        if t < 5:
            lambda_tar.append(0)
        elif t < 30 and t >= 5:
            lambda_tar.append((t - 5.0) / 25.0)
        else:
            lambda_tar.append(1)
    lambda_tar = np.array(lambda_tar)
    return torch.from_numpy(lambda_tar)

def write_to_bvhfile(data, filename, joints_to_remove):
    fout = open(filename, 'w')
    line_cnt = 0
    for line in open('./example.bvh', 'r'):
        fout.write(line)
        line_cnt += 1
        if line_cnt >= 132:
            break
    fout.write(('Frames: %d\n' % data.shape[0]))
    fout.write('Frame Time: 0.033333\n')
    pose_data = qeuler_np(data[:,3:].reshape(data.shape[0], -1, 4), order='zyx', use_gpu=False)
    # pose_data = np.concatenate([pose_data[:,:5], np.zeros_like(pose_data[:,0:1]),\
    #                             pose_data[:,5:9], np.zeros_like(pose_data[:,0:1]),\
    #                             pose_data[:,9:14], np.zeros_like(pose_data[:,0:1]),\
    #                             pose_data[:,14:18], np.zeros_like(pose_data[:,0:1]),\
    #                             pose_data[:,18:22], np.zeros_like(pose_data[:,0:1])], 1)
    pose_data = pose_data / np.pi * 180.0
    for t in range(data.shape[0]):
        line = '%f %f %f ' % (data[t, 0], data[t, 1], data[t, 2])
        for d in range(pose_data.shape[1] - 1):
            line += '%f %f %f ' % (pose_data[t, d, 2], pose_data[t, d, 1], pose_data[t, d, 0])
        line += '%f %f %f\n' % (pose_data[t, -1, 2], pose_data[t, -1, 1], pose_data[t, -1, 0])
        fout.write(line)
    fout.close()
        
    
