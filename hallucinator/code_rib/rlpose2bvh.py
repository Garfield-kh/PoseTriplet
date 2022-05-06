
import random
import argparse

import torch.optim as optim
import os
import datetime
import os.path as path
from torch.autograd import Variable

from progress.bar import Bar

from time import time
from bvh_skeleton import humanoid_rib_skeleton
from bvh_skeleton.camera import world2cam_sktpos

import torch
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing
import math



def rlpose2bvh(take_list, expert_dict):
    ##########################################################
    # save .
    # result_dict = {}
    takes = take_list
    result_all_dict = expert_dict

    write_standard_bvh_multi_process(takes, result_all_dict)
    ##########################################################
    return

def write_standard_bvh_multi_process(takes, result_all_dict):

    def wrap_write_standard_bvh(take):
        predicted_3d_wpos_withroot = np.copy(result_all_dict[take]['skt_wpos']).reshape(-1, 16, 3)
        # ground_z = np.min(predicted_3d_wpos_withroot[:, :, -1:])
        # ground_z = np.min(predicted_3d_wpos_withroot[:, :, -1:], axis=(1,2), keepdims=True)
        # predicted_3d_wpos_withroot[:, :, -1:] = predicted_3d_wpos_withroot[:, :, -1:] - ground_z

        bvhfileName = '{}/{}.bvh'.format(traj_save_path, take)
        write_standard_bvh(bvhfileName, predicted_3d_wpos_withroot)

    # start
    task_lst = takes
    # num_threads = args.num_threads

    for ep in range(math.ceil(len(task_lst) / num_threads)):

        p_lst = []
        for i in range(num_threads):
            idx = ep * num_threads + i
            if idx >= len(task_lst):
                break
            p = multiprocessing.Process(target=wrap_write_standard_bvh, args=(task_lst[idx],))
            p_lst.append(p)

        for p in p_lst:
            p.start()

        for p in p_lst:
            p.join()

        print('complete ep:', ep)
    # end.

def write_standard_bvh(bvhfileName, prediction3dpoint):
    '''
    :param outbvhfilepath:
    :param prediction3dpoint:
    :return:
    '''

    #
    prediction3dpoint = world2cam_sktpos(prediction3dpoint) * -1

    for frame in prediction3dpoint:
        for point3d in frame:
            point3d[0] *= 100
            point3d[1] *= 100
            point3d[2] *= 100

    mkd(bvhfileName)
    # 16 joint  21 joint
    Converter = humanoid_rib_skeleton.SkeletonConverter()
    prediction3dpoint = Converter.convert_to_22joint(prediction3dpoint)
    # bvh .
    human36m_skeleton = humanoid_rib_skeleton.H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)


    # ..mk dir
def mkd(target_dir, get_parent=True):
    # get parent path and create
    if get_parent:
        savedir = os.path.abspath(os.path.join(target_dir, os.pardir))
    else:
        savedir = target_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)


if __name__ == '__main__':
    """
    convert RL motion to rib motion
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_path', type=str, default='debug')
    args = parser.parse_args()

    pkl_path = args.pkl_path

    ######################################################################
    expert_dict = pickle.load(open(pkl_path, 'rb'))
    take_list = ['h36m_take_{:0>3d}'.format(i) for i in range(600)]
    num_threads = 32
    traj_save_path = './lafan1/lafan1'

    rlpose2bvh(take_list, expert_dict)


