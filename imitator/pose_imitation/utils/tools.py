import numpy as np
from utils.math import *


def normalize_traj(qpos_traj, qvel_traj):
    new_qpos_traj = []
    new_qvel_traj = []
    for qpos, qvel in zip(qpos_traj, qvel_traj):
        new_qpos = qpos.copy()
        new_qvel = qvel.copy()
        new_qvel[:3] = transform_vec(qvel[:3], qpos[3:7], 'heading')
        new_qpos[3:7] = de_heading(qpos[3:7])
        new_qpos_traj.append(new_qpos)
        new_qvel_traj.append(new_qvel)
    return np.vstack(new_qpos_traj), np.vstack(new_qvel_traj)


def sync_traj(qpos_traj, qvel_traj, ref_qpos):
    new_qpos_traj = []
    new_qvel_traj = []
    rel_heading = quaternion_multiply(get_heading_q(ref_qpos[3:7]), quaternion_inverse(get_heading_q(qpos_traj[0, 3:7])))
    ref_pos = ref_qpos[:3]
    start_pos = np.concatenate((qpos_traj[0, :2], ref_pos[[2]]))
    for qpos, qvel in zip(qpos_traj, qvel_traj):
        new_qpos = qpos.copy()
        new_qvel = qvel.copy()
        new_qpos[:2] = quat_mul_vec(rel_heading, qpos[:3] - start_pos)[:2] + ref_pos[:2]
        new_qpos[3:7] = quaternion_multiply(rel_heading, qpos[3:7])
        new_qvel[:3] = quat_mul_vec(rel_heading, qvel[:3])
        new_qpos_traj.append(new_qpos)
        new_qvel_traj.append(new_qvel)
    return np.vstack(new_qpos_traj), np.vstack(new_qvel_traj)


def remove_noisy_hands(results):
    for traj in results.values():
        for take in traj.keys():
            traj[take][..., 32:35] = 0
            traj[take][..., 42:45] = 0
    return


############################################################
############################################################
########################################
#  tools for GAN training
########################################
def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def get_discriminator_accuracy(prediction, label):
    '''
    this is to get discriminator accuracy for tensorboard
    input is tensor -> convert to numpy
    :param tensor_in: Bs x Score :: where score > 0.5 mean True.
    :return:
    '''
    # get numpy from tensor
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    rlt = np.abs(prediction - label)
    rlt = np.where(rlt > 0.5, 0, 1)
    num_of_correct = np.sum(rlt)
    accuracy = num_of_correct / label.shape[0]
    return accuracy


def root_2dpose(x):
    # b x 16 x 2
    sz = x.shape
    assert sz[1] == 16
    assert sz[2] == 2
    return x - x[:, :1]


