import torch.nn as nn
import torch
import numpy as np
import torchgeometry as tgm

'''
function on pose related information extraction.
'''


def get_pose3dbyBoneVec(bones, num_joints=16):
    '''
    conver bone vect to pose3d，is the inverse of get_bone_vector
    :param bones:
    :return:
    '''
    Ctinverse = torch.Tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 basement
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1],  # 14 15
    ]).transpose(1, 0)

    Ctinverse = Ctinverse.to(bones.device)
    C = Ctinverse.repeat([bones.size(0), 1, 1]).reshape(-1, num_joints - 1, num_joints)
    bonesT = bones.permute(0, 2, 1).contiguous()
    pose3d = torch.matmul(bonesT, C)
    pose3d = pose3d.permute(0, 2, 1).contiguous()  # back to N x 16 x 3
    return pose3d


def get_BoneVecbypose3d(x, num_joints=16):
    '''
    convert 3D point to bone vector
    :param x: N x number of joint x 3
    :return: N x number of bone x 3  number of bone = number of joint - 1
    '''
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 14 15
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), 1, 1]).reshape(-1, num_joints, num_joints - 1)
    pose3 = x.permute(0, 2, 1).contiguous()
    B = torch.matmul(pose3, C)
    B = B.permute(0, 2, 1)  # back to N x 15 x 3
    return B


def get_bone_lengthbypose3d(x, bone_dim=2):
    '''
    :param bone_dim: dim=2
    :return:
    '''
    bonevec = get_BoneVecbypose3d(x)
    bones_length = torch.norm(bonevec, dim=2, keepdim=True)
    return bones_length


def get_bone_unit_vecbypose3d(x, num_joints=16, bone_dim=2):
    bonevec = get_BoneVecbypose3d(x)
    bonelength = get_bone_lengthbypose3d(x)
    bone_unitvec = bonevec / bonelength
    return bone_unitvec



from function.utils import get_contacts
def get_leg_ratio(bl_old, bl_new):
    """
    bl: bx15x1
    """
    assert len(bl_old.shape) == 4
    assert len(bl_new.shape) == 4
    leg_len_old = bl_old[:, 0, 1, 0] + bl_old[:, 0, 2, 0]
    leg_len_new = bl_new[:, 0, 1, 0] + bl_new[:, 0, 2, 0]
    leg_ratio = leg_len_new / leg_len_old
    return leg_ratio  # bx1

def pose_seq_bl_aug(pose_seq_in, kbl=None):
    """
    kbl: kx15 from S15678 or s911
    pose_seq: tx16x3
    """
    if not kbl:
        kbl = np.load('./data/bonelength/bl_15segs_templates_mdifyed.npy').astype('float32')  # 15 bl from Evol(cvpr2020).
        kbl = torch.from_numpy(kbl)
    # size match
    pose_seq = pose_seq_in * 1.
    root_pose = pose_seq[:, :1, :] * 1.
    # get BL BV then convert back.
    i = np.random.randint(kbl.shape[0])
    bl = kbl[i] * 1.
    bl = bl.unsqueeze(0).unsqueeze(-1)
    bv = get_bone_unit_vecbypose3d(pose_seq)
    out = get_pose3dbyBoneVec(bv * bl)
    bl_old = get_bone_lengthbypose3d(pose_seq)
    leg_ratio = get_leg_ratio(bl_old.unsqueeze(0), bl.unsqueeze(0))
    out = out + root_pose * leg_ratio
    return out




def pose_seq_bl_reset(pose_seq_in):
    """
    pose_seq: tx16x3
    reset to RL bone length (average of s15678)
    """
    # size match
    pose_seq = pose_seq_in * 1.
    root_pose = pose_seq[:, :1, :] * 1.
    # bl = np.array([[[0.1332899], [0.4379], [0.447],
    #                [0.1332899], [0.4379], [0.447],
    #                [0.24004446], [0.2710998], [0.16976325],
    #                [0.15269038], [0.2798], [0.25],
    #                [0.15269038], [0.2798], [0.25]]], dtype='float32')
    bl = np.array([[[0.13545841], [0.45170274], [0.4469572],
                   [0.13545777], [0.45170122], [0.44695726],
                   [0.2414928], [0.25551477], [0.18441138],
                   [0.15050778], [0.28198972], [0.24994883],
                   [0.15050682], [0.28199276], [0.24994786]]], dtype='float32')  # bone length used in RL
    bl = torch.from_numpy(bl)
    bv = get_bone_unit_vecbypose3d(pose_seq)
    out = get_pose3dbyBoneVec(bv * bl)
    bl_old = get_bone_lengthbypose3d(pose_seq)
    leg_ratio = get_leg_ratio(bl_old.unsqueeze(0), bl.unsqueeze(0))
    out = out + root_pose * leg_ratio
    return out



def pose_seq_bl_aug_batch(pose_seq_batch):
    """
    pose_seq_batch: b x t x j x c
    """
    b, t, j, c = pose_seq_batch.shape
    # s15678bl  5x15
    # kbl = np.load('./data/bonelength/hm36s15678_bl_templates.npy')
    kbl = np.load('./data/bonelength/bl_15segs_templates_mdifyed.npy').astype('float32')  # 15 bl from Evol git.
    kbl = torch.from_numpy(kbl.astype('float32')).to(pose_seq_batch.device)
    # random b bl
    bbl_idx = np.random.choice(kbl.shape[0], b)
    bbl = kbl[bbl_idx]  # b x 15
    bbl = bbl.unsqueeze(1).unsqueeze(-1)  # bx1x15x1
    # root traj
    root_pose = pose_seq_batch[:, :, :1, :] * 1.
    pose_seq_bt = pose_seq_batch.reshape(b*t, 16, 3) * 1.
    # calculate bl
    bl_old_bt = get_bone_lengthbypose3d(pose_seq_bt) # bt x 15 x 1
    bl_old = bl_old_bt.reshape(b, t, 15, 1)
    # calculate ratio
    leg_ratio = get_leg_ratio(bl_old, bbl).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # bx1x1x1

    # change BL
    bv_bt = get_bone_unit_vecbypose3d(pose_seq_bt)
    bv = bv_bt.reshape(b, t, 15, 3)
    bv = bv * bbl
    bv_bt = bv.reshape(b*t, 15, 3)
    out_bt = get_pose3dbyBoneVec(bv_bt)
    out = out_bt.reshape(b, t, 16, 3)
    out = out + root_pose * leg_ratio
    return out



def kcs_layer_unit(x, num_joints=16):
    # implementation of the Kinematic Chain Space as described in the paper
    # apply local KCS later. by mask the Ct.
    # KCS matrix
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 14 15
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), 1, 1]).reshape(-1, num_joints, num_joints - 1)
    x = x.reshape(x.size(0), -1, 3)
    pose3 = x.permute(0, 2, 1).contiguous()  # 这里16x3变成3x16的话 应该用permute吧
    B = torch.matmul(pose3, C)
    B = B / torch.norm(B, dim=1, keepdim=True)
    Psi = torch.matmul(B.permute(0, 2, 1), B)

    return Psi


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



# basic tool

def diff(input, axis=None):
    # now is b t j 3
    tmp = input[:, 1:] - input[:, :-1]
    return torch.cat([tmp, tmp[:, -1:]], dim=1)


import copy


# To store 50 generated image in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=4096):
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


def diff_range_loss(a, b, std):
    diff = (torch.abs(a) - b) ** 2
    diff = torch.mean(diff, dim=-1, keepdim=True)
    weight = torch.where(diff > std ** 2, torch.ones_like(diff), torch.zeros_like(diff))
    diff_weighted = diff * weight
    return diff_weighted.mean()


#######
def btjd2bft(x):
    '''
    convert bxtxjx3 to b x j x t for 1D conv
    '''
    assert len(x.shape) == 4
    assert x.shape[-2] == 16

    sz = x.shape
    x = x.view(sz[0], sz[1], -1)
    x = x.permute(0, 2, 1)
    return x


def bft2btjd(x):
    '''
    convert bxtxjx3 to b x j x t for 1D conv
    '''
    assert len(x.shape) == 3

    sz = x.shape
    x = x.permute(0, 2, 1)
    x = x.view(sz[0], sz[2], 16, -1)
    return x
