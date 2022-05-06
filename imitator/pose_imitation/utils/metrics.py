from utils import *
from utils.transformation import euler_from_quaternion


def get_joint_angles(poses):
    root_angs = []
    for pose in poses:
        root_euler = np.array(euler_from_quaternion(pose[3:7]))
        root_euler[2] = 0.0
        root_angs.append(root_euler)
    root_angs = np.vstack(root_angs)
    angles = np.hstack((root_angs, poses[:, 7:]))
    return angles


def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i+1], dt, 'heading')
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_mean_abs(x):
    return np.abs(x).mean()


