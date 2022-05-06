
import random

import torch.optim as optim
import os
import datetime
import os.path as path
from torch.autograd import Variable

from progress.bar import Bar

from time import time
from bvh_skeleton import humanoid_1205_skeleton
from bvh_skeleton.camera import world2cam_sktpos, cam2world_sktpos

import torch
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing
import math

import glob

import argparse


def ribpose2bvh(take_list, expert_dict):
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
    # prediction3dpoint = world2cam_sktpos(prediction3dpoint) * -1
    prediction3dpoint = cam2world_sktpos(prediction3dpoint * -1)



    mkd(bvhfileName)
    # 16 joint 21 joint
    Converter = humanoid_1205_skeleton.SkeletonConverter()
    prediction3dpoint = Converter.convert_to_21joint(prediction3dpoint)
    #  bvh .
    human36m_skeleton = humanoid_1205_skeleton.H36mSkeleton()
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
    convert rib motion to rl motion
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_save_path', type=str, default='debug')
    args = parser.parse_args()

    traj_save_path = args.traj_save_path


    npy_folder = '../traj_pose'
    npy_path_list = glob.glob(npy_folder+'/*.npy')
    array_list = []
    for npy_path in npy_path_list:
        tmp = np.load(npy_path)
        array_list.append(tmp)

    rib_pose_seq = np.concatenate(array_list, axis=0)

    # add pose in dict.
    take_list = ['h36m_take_{:0>3d}'.format(i) for i in range(rib_pose_seq.shape[0] + 600)][600:]
    expert_dict = {}
    #convert pose 22 joint to 16joint
    joint_keep = [0,1,2,3,5,6,7,11,12,13,15,16,17,19,20,21]
    for i, take in enumerate(take_list):
        expert_dict[take] = {}
        tmp_1 = rib_pose_seq[i] * 1.
        tmp_2 = tmp_1[1:, joint_keep, :]
        expert_dict[take]['skt_wpos'] = tmp_2


    ######################################################################
    num_threads = 12

    ribpose2bvh(take_list, expert_dict)


