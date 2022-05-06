from __future__ import absolute_import, division

import os
import torch
import numpy as np
from tensorboardX import SummaryWriter


# self define tools
class Summary(object):
    def __init__(self, directory):
        self.directory = directory
        self.epoch = 0
        self.writer = None
        self.phase = 0
        self.train_iter_num = 0
        self.train_realpose_iter_num = 0
        self.train_fakepose_iter_num = 0
        self.train_realtraj_iter_num = 0
        self.train_faketraj_iter_num = 0
        self.test_iter_num = 0
        self.test_MPI3D_iter_num = 0

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def summary_train_iter_num_update(self):
        self.train_iter_num = self.train_iter_num + 1

    def summary_train_realpose_iter_num_update(self):
        self.train_realpose_iter_num = self.train_realpose_iter_num + 1

    def summary_train_fakepose_iter_num_update(self):
        self.train_fakepose_iter_num = self.train_fakepose_iter_num + 1

    def summary_train_realtraj_iter_num_update(self):
        self.train_realtraj_iter_num = self.train_realtraj_iter_num + 1

    def summary_train_faketraj_iter_num_update(self):
        self.train_faketraj_iter_num = self.train_faketraj_iter_num + 1

    def summary_test_iter_num_update(self):
        self.test_iter_num = self.test_iter_num + 1

    def summary_test_MPI3D_iter_num_update(self):
        self.test_MPI3D_iter_num = self.test_MPI3D_iter_num + 1

    def summary_epoch_update(self):
        self.epoch = self.epoch + 1

    def summary_phase_update(self):
        self.phase = self.phase + 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from torch.optim import lr_scheduler
'cp from dlow'
def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


########################################
#  generate mask for pose sequence input
########################################
MASK_MODES = ('No mask', 'Future Prediction', 'Missing Frames', 'Structured Occlusion')

body_members = {
            'left_arm': {'joints': [8, 9, 10, 11], 'side': 'left'},
            'right_arm': {'joints': [8, 12, 13, 14], 'side': 'right'},
            'head': {'joints': [8, 9], 'side': 'right'},
            'torso': {'joints': [0, 7, 8], 'side': 'right'},
            'left_leg': {'joints': [0, 4, 5, 6], 'side': 'left'},
            'right_leg': {'joints': [0, 1, 2, 3], 'side': 'right'},
        }


def gen_mask(mask_type, keep_prob, batch_size, njoints, seq_len, body_members=body_members, keep_feet=False):
    # Default mask, no mask
    mask = np.ones(shape=(batch_size, seq_len, njoints, 1))
    if mask_type == 1:  # Future Prediction
        mask[:, np.int(seq_len * keep_prob):, :, :] = 0.0
    elif mask_type == 2:  # Missing Frames patch
        occ_frames = np.random.randint(seq_len - np.int(seq_len * keep_prob), size=1)
        mask[:, np.int(occ_frames):np.int(occ_frames+np.int(seq_len * keep_prob)), :, :] = 0.0
    elif mask_type == 3:  # Structured Occlusion Simulation
        rand_joints = set()
        while ((njoints - len(rand_joints)) >
               (njoints * keep_prob)):
            joints_to_add = (list(body_members.values())[np.random.randint(len(body_members))])['joints']
            for joint in joints_to_add:
                rand_joints.add(joint)
        mask[:, :, list(rand_joints), :] = 0.0

    if keep_feet and np.random.uniform() > 0.5:
        # keep feet
        mask[:, :, [3, 6], :] = 1.0
    # This unmasks first and last frame for all sequences (required for baselines)
    # all should have, to avoid static prediction
    mask[:, [0, -1], :, :] = 1.0
    return mask

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

import copy
# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=None):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


import numpy as np

def get_contacts(poses):
    '''
    https://github.com/Shimingyi/MotioNet/blob/fbceb5ffa85a509ed5b42b06c1766cea9cdcd328/data/h36m_dataset.py
    pose contact label extraction
    :param poses:
    :return:
    '''
    poses_reshape = poses.reshape((-1, 16, 3))
    contact_signal = np.zeros((poses_reshape.shape[0], 3))
    left_z = poses_reshape[:, 3, 2]
    right_z = poses_reshape[:, 6, 2]

    contact_signal[left_z <= (np.mean(np.sort(left_z)[:left_z.shape[0] // 5]) + 2e-2), 0] = 1
    contact_signal[right_z <= (np.mean(np.sort(right_z)[:right_z.shape[0] // 5]) + 2e-2), 1] = 1
    left_velocity = np.sqrt(np.sum((poses_reshape[2:, 3] - poses_reshape[:-2, 3]) ** 2, axis=-1))
    right_velocity = np.sqrt(np.sum((poses_reshape[2:, 6] - poses_reshape[:-2, 6]) ** 2, axis=-1))
    contact_signal[1:-1][left_velocity >= 5e-3, 0] = 0
    contact_signal[1:-1][right_velocity >= 5e-3, 1] = 0
    return contact_signal


def check_isNone(cklst):
    """
    cklst: list of tensor
    :return:
    """
    for item in cklst:
        assert not torch.isnan(item).any()

	# ..mk dir
def mkd(target_dir, get_parent=True):
    # get parent path and create
    if get_parent:
        savedir = os.path.abspath(os.path.join(target_dir, os.pardir))
    else:
        savedir = target_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

