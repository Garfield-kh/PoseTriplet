# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torchgeometry as tgm

from common.utils import wrap
from common.quaternion import qrot, qinverse

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    
####################################
# world_to_camera camera_to_world
# X:nxjx3 R:4 t:3
####################################
def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

####################################
# world_to_camera camera_to_world in torch version
# X:nxjx3 R:nx4 t:nx3
####################################
def world_to_cameraByTensor(X, R, t):
    Rt = qinverse(R) # Invert rotation
    return qrot(Rt.unsqueeze(1).unsqueeze(1).repeat(1, *X.shape[1:-1], 1), X - t.unsqueeze(1).unsqueeze(1).repeat(1, *X.shape[1:-1], 1)) # Rotate and translate

    # tmp_X = X.view(-1, 16, 3)
    # tmp_R = R.view(-1, 4)
    # tmp_t = t.view(-1, 3)
    # tmp_Rt = qinverse(tmp_R)  # Invert rotation
    # tmp_out = qrot(tmp_Rt, tmp_X-tmp_t) # Rotate and translate
    # return tmp_out.view(X.shape)

def camera_to_worldByTensor(X, R, t):
    # tmp = R.unsqueeze(1).unsqueeze(1).repeat(1, *X.shape[1:-1], 1)
    # tmp2 = qrot(R.unsqueeze(1).unsqueeze(1).repeat(1, *X.shape[1:-1], 1), X)
    return qrot(R.unsqueeze(1).unsqueeze(1).repeat(1, *X.shape[1:-1], 1), X) + t.unsqueeze(1).unsqueeze(1).repeat(1, *X.shape[1:-1], 1)

    # tmp_X = X.view(-1, 16, 3)
    # tmp_R = R.view(-1, 4)
    # tmp_t = t.view(-1, 3)
    # tmp_out = qrot(tmp_R, tmp_X) + tmp_t # Rotate and translate
    # return tmp_out.view(X.shape)


################################

def world2cam_sktpos(skt_in):
    # t1 = skt_in.reshape(-1, 17, 3)
    # gt_3d_world = np.delete(t1, (8), axis=1)  # remove nose
    gt_3d_world = skt_in.reshape(-1, 16, 3)
    # from x y z to x z -y
    gt_3d = gt_3d_world[..., [0, 2, 1]]
    # gt_3d[:, :, 1] = gt_3d[:, :, 1] * -1
    gt_3d = gt_3d * -1
    return gt_3d

def cam2world_sktpos(skt_in):
    # change Z up to Y up coordinate
    # skint_in: b t j 3
    tmp_skt_in = skt_in * 1.0
    # tmp_skt_in[:, :, :, 1] = tmp_skt_in[:, :, :, 1] * -1
    tmp_skt_in = tmp_skt_in * -1
    gt_3d_world = tmp_skt_in[..., [0, 2, 1]]
    return gt_3d_world

################################

####################################
# world_to_camera camera_to_world
# apply for sktpos
# X:nxjx3 R:4 t:3
####################################
def reset_center(tmp):
    """
    tmp:tx16x3
    """
    tmp = tmp * 1.
    x = tmp[:, 0, 0]
    y = tmp[:, 0, 1]
    xmin = np.min(x)
    xmax = np.max(x)
    xcenter = (xmin + xmax) * 0.5
    ymin = np.min(y)
    ymax = np.max(y)
    ycenter = (ymin + ymax) * 0.5 - 0.4
    offset = np.array([[[xcenter, ycenter, 0]]])
    return tmp - offset


def set_center_v2(tmp):
    """
    tmp:tx16x3

    """
    x = tmp[:, 0, 0]
    y = tmp[:, 0, 1]
    xmin = np.min(x)
    xmax = np.max(x)
    xcenter = (xmin + xmax) * 0.5
    ymin = np.min(y)
    ymax = np.max(y)
    ycenter = (ymin + ymax) * 0.5
    center = np.array([[[xcenter, ycenter, 0]]])

    # add some random placement.
    w = xmax - xmin
    h = ymax - ymin
    static = 1
    stat = 0.4
    if w < static and h < static:
        x_offset = np.random.uniform(-stat, +stat)
        y_offset = np.random.uniform(-stat, +stat) + 0.4
    else:
        x_offset = 0
        y_offset = 0.4
    offset = np.array([[[x_offset, y_offset, 0]]])

    return tmp - center + offset


def world_to_camera_sktpos_v2(X, R, t):
    # X = reset_center(X).astype('float32')
    X = set_center_v2(X).astype('float32')
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate

from scipy.spatial.transform import Rotation as R
def world_to_camera_sktpos_v3(X, args):
    """
    random a camera around the person for projection
    """
    def wxyz2xyzw(wfist):
        "convert w x y z to x y z w, xyzw is used in scipy."
        return np.stack([wfist[1], wfist[2], wfist[3], wfist[0]], axis=0)

    def xyzw2wxyz(wlast):
        "convert x y z w to w x y z, wxyz is used in qrot"
        return np.stack([wlast[3], wlast[0], wlast[1], wlast[2]], axis=0)

    posi_x = np.random.uniform(args.rpx_min, args.rpx_max)
    posi_y = 0
    posi_z = np.random.uniform(args.rpz_min, args.rpz_max)
    cam_p = np.array([posi_x, posi_y, posi_z]).astype('float32')

    euler_x = np.random.uniform(args.rex_min, args.rex_max)
    euler_y = np.random.uniform(args.rey_min, args.rey_max)
    euler_z = np.random.uniform(args.rez_min, args.rez_max)
    cam_r = R.from_euler('xyz', [euler_x, euler_y, euler_z], degrees=True)
    cam_q = cam_r.as_quat()
    cam_q = xyzw2wxyz(cam_q).astype('float32')

    # X = reset_center(X).astype('float32')
    X = set_center_v2(X).astype('float32')
    Rt = wrap(qinverse, cam_q)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - cam_p)  # Rotate and translate


def world_to_camera_sktpos_v3_new(X, args):
    """
    random a camera around the person for projection
    try solving unstable projection training - change to norm sample, or clip more gridient
    """
    def norm_sample_withbound(lb, ub):
        mu = 0.5 * (lb + ub)
        sigma = 0.3 * (ub - lb)
        s = np.random.normal(mu, sigma)

        if s < lb or s > ub:
            s = norm_sample_withbound(lb, ub)
        return s

    def wxyz2xyzw(wfist):
        "convert w x y z to x y z w, xyzw is used in scipy."
        return np.stack([wfist[1], wfist[2], wfist[3], wfist[0]], axis=0)

    def xyzw2wxyz(wlast):
        "convert x y z w to w x y z, wxyz is used in qrot"
        return np.stack([wlast[3], wlast[0], wlast[1], wlast[2]], axis=0)

    posi_x = norm_sample_withbound(args.rpx_min, args.rpx_max)
    posi_y = 0
    posi_z = norm_sample_withbound(args.rpz_min, args.rpz_max)
    cam_p = np.array([posi_x, posi_y, posi_z]).astype('float32')

    euler_x = norm_sample_withbound(args.rex_min, args.rex_max)
    euler_y = norm_sample_withbound(args.rey_min, args.rey_max)
    euler_z = norm_sample_withbound(args.rez_min, args.rez_max)
    cam_r = R.from_euler('xyz', [euler_x, euler_y, euler_z], degrees=True)
    cam_q = cam_r.as_quat()
    cam_q = xyzw2wxyz(cam_q).astype('float32')

    # X = reset_center(X).astype('float32')
    X = set_center_v2(X).astype('float32')
    Rt = wrap(qinverse, cam_q)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - cam_p)  # Rotate and translate


def zaxis_randrotation(x_in):
    '''
    x: t j 3
    '''
    # x_root = x[:1, :1, :] * 1.0
    # x_rooted = x - x_root
    #
    # random_z = 6.28 * np.random.uniform(0, 1, (1,3)).astype('float32')
    # random_z[:, :3] = 0
    # random_qz = tgm.angle_axis_to_quaternion(torch.from_numpy(random_z)).numpy().astype('float32')
    # x_rooted = camera_to_world(x_rooted, random_qz, np.zeros_like(random_z))

    x = torch.from_numpy(x_in).unsqueeze(0)  # 1 t j c
    x_rooted = posegan_preprocess(x)
    return x_rooted.numpy()[0]

def posegan_preprocess(x, aug_rotate=True):
    '''
    x: b t j 3
    '''
    x_root = x[:, :1, :1, :] * 1.0
    x_rooted = x - x_root
    if aug_rotate:
        random_z = 6.28 * torch.rand(x.shape[0], 3).to(x.device)  # b x 7
        random_z[:, :2] = 0
        # random_z[:, :] = 0
        random_qz = tgm.angle_axis_to_quaternion(random_z)
        x_rooted = camera_to_worldByTensor(x_rooted, random_qz, torch.zeros_like(random_z))
    return x_rooted + x_root

#############################################################

# def world_to_cameraByTensor(X, R, t):
#     tmp_X, tmp_R, tmp_t = X.detach().cpu().numpy(), R.detach().cpu().numpy(), t.detach().cpu().numpy()
#     tmp_Rt = wrap(qinverse, tmp_R) # Invert rotation
#     out = wrap(qrot, np.tile(tmp_Rt, (1, *tmp_X.shape[1:-1], 1)), tmp_X - tmp_t) # Rotate and translate
#
# def camera_to_worldByTensor(X, R, t):
#     tmp_X, tmp_R, tmp_t = X.detach().cpu().numpy(), R.detach().cpu().numpy(), t.detach().cpu().numpy()
#     out = wrap(qrot, np.tile(tmp_R, (1, *tmp_X.shape[:-1], 1)), tmp_X) + tmp_t
#     out = torch.from_numpy(out)
#     return out.to(X.device)

####################################
# project to 2D in torch version
# supposed workable for any size x J x 3
####################################
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.sh.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f*XX + c


def project_to_2d_purelinear(X):
    """
    Project 3D points to 2D using only linear parameters .

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    """
    assert X.shape[-1] == 3

    # XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    XX = X[..., :2] / X[..., 2:]

    # f is the scale that related to the absolute depth information.
    f = 2.3

    return XX * f

