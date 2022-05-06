import os
import pickle
import argparse
from scipy.ndimage.filters import median_filter


# ..mk dir
def mkd(target_dir, get_parent=True):
    # get parent path and create
    if get_parent:
        savedir = os.path.abspath(os.path.join(target_dir, os.pardir))
    else:
        savedir = target_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

#############################################################################
# section: math 3d
#############################################################################

def dcm2quat(dcm):
    q = np.zeros([4])
    tr = np.trace(dcm)

    if tr > 0:
        sqtrp1 = np.sqrt(tr + 1.0)
        q[0] = 0.5 * sqtrp1
        q[1] = (dcm[1, 2] - dcm[2, 1]) / (2.0 * sqtrp1)
        q[2] = (dcm[2, 0] - dcm[0, 2]) / (2.0 * sqtrp1)
        q[3] = (dcm[0, 1] - dcm[1, 0]) / (2.0 * sqtrp1)
    else:
        d = np.diag(dcm)
        if d[1] > d[0] and d[1] > d[2]:
            sqdip1 = np.sqrt(d[1] - d[0] - d[2] + 1.0)
            q[2] = 0.5 * sqdip1

            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1

            q[0] = (dcm[2, 0] - dcm[0, 2]) * sqdip1
            q[1] = (dcm[0, 1] + dcm[1, 0]) * sqdip1
            q[3] = (dcm[1, 2] + dcm[2, 1]) * sqdip1

        elif d[2] > d[0]:
            sqdip1 = np.sqrt(d[2] - d[0] - d[1] + 1.0)
            q[3] = 0.5 * sqdip1

            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1

            q[0] = (dcm[0, 1] - dcm[1, 0]) * sqdip1
            q[1] = (dcm[2, 0] + dcm[0, 2]) * sqdip1
            q[2] = (dcm[1, 2] + dcm[2, 1]) * sqdip1

        else:
            sqdip1 = np.sqrt(d[0] - d[1] - d[2] + 1.0)
            q[1] = 0.5 * sqdip1

            if sqdip1 != 0:
                sqdip1 = 0.5 / sqdip1

            q[0] = (dcm[1, 2] - dcm[2, 1]) * sqdip1
            q[2] = (dcm[0, 1] + dcm[1, 0]) * sqdip1
            q[3] = (dcm[2, 0] + dcm[0, 2]) * sqdip1

    return q


def quat_divide(q, r):
    return quat_mul(quat_inverse(r), q)

def quat_mul(q0, q1):
    original_shape = q0.shape
    q1 = np.reshape(q1, [-1, 4, 1])
    q0 = np.reshape(q0, [-1, 1, 4])
    terms = np.matmul(q1, q0)
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    q_product = np.stack([w, x, y, z], axis=1)
    return np.reshape(q_product, original_shape)


def quat_inverse(q):
    original_shape = q.shape
    q = np.reshape(q, [-1, 4])

    q_conj = [q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]
    q_conj = np.stack(q_conj, axis=1)
    q_inv = np.divide(q_conj, quat_dot(q_conj, q_conj))

    return np.reshape(q_inv, original_shape)

def quat2euler(q, order='zxy', eps=1e-8):
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = np.reshape(q, [-1, 4])

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'zxy':
        x = np.arcsin(np.clip(2 * (q0 * q1 + q2 * q3), -1 + eps, 1 - eps))
        y = np.arctan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        euler = np.stack([z, x, y], axis=1)
    else:
        raise ValueError('Not implemented')

 # todo adding order xyz
    if order == 'zxy':
        x = np.arcsin(np.clip(2 * (q0 * q1 + q2 * q3), -1 + eps, 1 - eps))
        y = np.arctan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        euler = np.stack([z, x, y], axis=1)
    else:
        raise ValueError('Not implemented')

    return np.reshape(euler, original_shape)



def quat_dot(q0, q1):
    original_shape = q0.shape
    q0 = np.reshape(q0, [-1, 4])
    q1 = np.reshape(q1, [-1, 4])

    w0, x0, y0, z0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    q_product = w0 * w1 + x1 * x1 + y0 * y1 + z0 * z1
    q_product = np.expand_dims(q_product, axis=1)
    q_product = np.tile(q_product, [1, 4])

    return np.reshape(q_product, original_shape)


#############################################################################
# section: random curve generation
#############################################################################

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

# def ccw_sort(p):
#     d = p-np.mean(p,axis=0)
#     s = np.arctan2(d[:,0], d[:,1])
#     return p[np.argsort(s),:]

def ccw_sort(p):
    return p

def get_bezier_curve(a, rad=0.2, edgy=0, **kw):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var", **kw)
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


def get_random_traj(n=4, scale=1, rad=0.3, edgy=0.05, numpoints=100, random_flip=False):
    """
    rad=0.3, edgy=0.05;
    rad=10, edgy=0.0
    :param n: total point
    :param scale: scale map
    :param numpoints: num of point for each
    :return:
    """
    a = get_random_points(n, scale)
    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy, numpoints=numpoints)
    if random_flip and np.random.randn() > 0:
        y = - y
    return np.stack([x,y]).T


# if __name__ == '__main__':
#     fig, ax = plt.subplots()
#     ax.set_aspect("equal")

#     rad = 10
#     edgy = 100

#     a = get_random_points(n=10, scale=1)
#     x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
#     plt.plot(x, y)

#     plt.show()

def get_straight_path(scale):
    x = np.linspace(0, scale, 500)
    y = np.zeros_like(x)
    return np.stack([x, y]).T



def get_rand_with_range(size, lb, ub):
    return np.random.rand(size) * (ub - lb) + lb


def get_hip_z(trajxy):
    """add som cps."""
    size = trajxy.shape[0]
    dist = np.arange(size)
    T = get_rand_with_range(1, 25, 50)
    H = get_rand_with_range(1, 0.95, 1.1)
    A = get_rand_with_range(1, 0.05, 0.08)

    # A 0.03 T 3/5, H 0.835
    hipz = np.cos(2 * np.pi / T * dist) * A + H
    # hipz = hipz - np.min(hipz) - 0.15 + 1. + 0.1
    hipz = hipz - np.min(hipz) - 0.15 + 1. + 0.0
    return hipz

from scipy.spatial.transform import Rotation as R

def get_random_qpos(plot=True, scale=0.4, random_flip=True, curve_type='curve'):
    if curve_type=='curve':
        trajxy = get_random_traj(n=5, scale=scale, rad=0.3, edgy=0.05, numpoints=400, random_flip=random_flip)[:600]   # 500 +  默认就用500的.
    elif curve_type == 'circle':
        trajxy = get_random_traj(n=6, scale=scale, rad=0.5, edgy=0.0, numpoints=100, random_flip=True)[
                 :600]
    elif curve_type=='straight':
        trajxy = get_straight_path(scale=scale)
    else:
        assert False, 'unkonw curve_type: {}'.format(curve_type)
    print('trajxy.shape: ', trajxy.shape)

    if plot:
        # plt.plot(trajxy[:,0],  trajxy[:,1], '-', c='g', linewidth=0.5, markersize=0.1)
        # plt.show()
        plt.plot(trajxy[:500,0],  trajxy[:500,1], '.',  c='b', linewidth=0.5, markersize=0.5)
        plt.plot(trajxy[:100,0],  trajxy[:100,1], 'x',  c='r', linewidth=0.5, markersize=0.5)
        plt.show()

        vtraj_root = trajxy[:-1] - trajxy[1:]

        vm = np.linalg.norm(vtraj_root, axis=-1)
        _ = plt.hist(vm, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram " + 'scale {}'.format(str(scale)))
        plt.show()

    """
    add hip height
    """
    hh = np.ones_like(trajxy) * 0.85
    trajxyz = np.concatenate([trajxy, hh[:,:1]], axis=-1)

    "vel"
    velxyz = trajxyz[1:] - trajxyz[:-1]
    velxyz = np.concatenate([velxyz, velxyz[-1:]], axis=0)
    hip_dir = np.cross(velxyz, np.array([0,0,1]))

    # if plot:
    #     # hip_dir
    #     print(velxyz[0])
    #     print(hip_dir[0])
    #
    #     plt.scatter(hip_dir[0, 0], hip_dir[0, 1])
    #     plt.scatter(velxyz[0, 0], velxyz[0, 1])
    #     plt.scatter(0, 0)
    #     plt.show()


    """
    add hip orientation
    """

    x_dir = -hip_dir / (np.linalg.norm(hip_dir, axis=-1, keepdims=True) + 1e-12)
    y_dir = -velxyz  / (np.linalg.norm(velxyz, axis=-1, keepdims=True) + 1e-12)
    z_dir = np.array([0,0,1])

    # hip_dir
    print('x_dir[0] ', x_dir[0])
    print('y_dir[0]', y_dir[0])

    # if plot:
    #     plt.scatter(x_dir[0, 0], x_dir[0, 1])
    #     plt.scatter(y_dir[0, 0], y_dir[0, 1])
    #     plt.scatter(0, 0)
    #     plt.show()

    order = 'zxy'

    hip_euler = []
    hip_local_quat = []
    for i in range(x_dir.shape[0]):
        dcm_hip = np.asarray([x_dir[i], y_dir[i], z_dir])
        quat_hip = dcm2quat(dcm_hip)

        dcm_world = np.asarray([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
        quat_world = dcm2quat(dcm_world)

        local_quat = quat_divide(
            q=quat_hip, r=quat_world
        )


        hip_local_quat.append(local_quat)

    # hip_euler = np.stack(hip_euler)
    # hip_euler = hip_euler[:, [1, 2, 0]]
    # print('hip_euler.shape ', hip_euler.shape)

    hip_local_quat = np.stack(hip_local_quat)
    print('hip_local_quat.shape ', hip_local_quat.shape)

    inputs_2d = np.random.randn(500, 16, 2)
    # predicted_3d_qpos = np.random.randn(500, 59)
    predicted_3d_qpos = np.zeros((500, 59))

    hh = get_hip_z(trajxy)
    hh = np.expand_dims(hh, -1)
    trajxyz = np.concatenate([trajxy, hh[:,:1]], axis=-1)

    predicted_3d_qpos[:, :3] = trajxyz[:500]
    predicted_3d_qpos[:, 3:7] = hip_local_quat[:500]

    return {
        'inputs_2d': inputs_2d,
        'predicted_3d_qpos': predicted_3d_qpos,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale-start', type=float, default='1')
    parser.add_argument('--scale-end', type=float, default='20')
    parser.add_argument('--num-take', type=int, default='20')
    parser.add_argument('--curve-type', type=str, default='curve')  # curve / straight
    parser.add_argument('--mocap-folder', type=str, default='./checkpoint/exp_rcs_debug/helix_0')
    args = parser.parse_args()
    ##########################################################
    # save .
    # result_dict = {}
    takes = ['h36m_take_{:0>3d}'.format(i) for i in range(args.num_take)]
    scale_list = np.linspace(args.scale_start, args.scale_end, args.num_take)
    result_all_dict = {}
    for i, take in enumerate(takes):
        scale = scale_list[i]
        result_all_dict[take] = get_random_qpos(plot=False, scale=scale, random_flip=True, curve_type=args.curve_type)
    traj_save_path = '{}/datasets/traj_dict/traj_dict.pkl'.format(args.mocap_folder)
    mkd(traj_save_path)
    with open(traj_save_path, 'wb') as f:
        pickle.dump(result_all_dict, f)
