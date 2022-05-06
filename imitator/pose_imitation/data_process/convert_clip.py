import os
import sys

sys.path.append(os.getcwd())

from utils import *
from utils.transformation import quaternion_from_euler
from mujoco_py import load_model_from_path
from mocap.skeleton import Skeleton
from mocap.pose import load_bvh_file, interpolated_traj
import pickle
import glob
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=True)
parser.add_argument('--model-id', type=str, default='humanoid_1205_orig')
parser.add_argument('--mocap-id', type=str, default='0626')  # default: 0213
parser.add_argument('--range', type=int, default=None)  # default: (5, 20)
parser.add_argument('--skt-id', type=str, default='take_000')
parser.add_argument('--mocap-fr', type=int, default=30)  # default 120
parser.add_argument('--dt', type=float, default=1/30)
parser.add_argument('--offset-z', type=float, default=0.0)
args = parser.parse_args()

model_file = 'assets/mujoco_models/%s.xml' % args.model_id
model = load_model_from_path(model_file)
body_qposaddr = get_body_qposaddr(model)

skt_bvh = os.path.expanduser('datasets/traj_debug/%s_%s.bvh' % (args.mocap_id, args.skt_id))
exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                 'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
skeleton = Skeleton()
skeleton.load_from_bvh(skt_bvh, exclude_bones, spec_channels)


def get_qpos(pose, bone_addr):
    """

    :param pose:   - ind1 = [a, b]  ind2 = [a', b']
    :param bone_addr: bvh address
    :return:
    """
    qpos = np.zeros(model.nq)
    for bone_name, ind2 in body_qposaddr.items():
        ind1 = bone_addr[bone_name]
        if ind1[0] == 0:
            trans = pose[ind1[0]:ind1[0] + 3].copy()
            angles = pose[ind1[0] + 3:ind1[1]].copy()
            quat = quaternion_from_euler(angles[0], angles[1], angles[2], 'rxyz')
            qpos[ind2[0]:ind2[0] + 3] = trans
            qpos[ind2[0] + 3:ind2[1]] = quat
        else:
            qpos[ind2[0]:ind2[1]] = pose[ind1[0]:ind1[1]]
    return qpos


def get_poses(bvh_file):
    time0_get_poses = time.time()  # time start load.

    poses, bone_addr = load_bvh_file(bvh_file, skeleton)
    poses_samp = interpolated_traj(poses, args.dt, mocap_fr=args.mocap_fr)
    qpos_traj = []
    for i in range(poses_samp.shape[0]):
        cur_pose = poses_samp[i, :]
        cur_qpos = get_qpos(cur_pose, bone_addr)
        qpos_traj.append(cur_qpos)
    qpos_traj = np.vstack(qpos_traj)
    qpos_traj[:, 2] += args.offset_z

    time_cost_get_poses = time.time() - time0_get_poses  # time spend.
    print('-----> get_poses spends {:.2f}s on {} with {:0>6d} frames'.format(time_cost_get_poses,
                                                                             bvh_file, poses.shape[0]))
    return qpos_traj

bvh_files = glob.glob(os.path.expanduser('datasets/traj_debug/%s_*.bvh' % args.mocap_id))
bvh_files.sort()
if args.range is not None:
    bvh_files = bvh_files[args.range[0]: args.range[1]]
print('bvh_files:', bvh_files)

tmp_dict = {}
for file in bvh_files:
    print('extracting trajectory from %s' % file)
    qpos_traj = get_poses(file)
    name = os.path.splitext(os.path.basename(file))[0]
    tmp_dict[name] = {}
    tmp_dict[name]['predicted_3d_qpos'] = qpos_traj


# with open('./datasets/traj_new/traj_dict_ego.pkl', 'wb') as f:
#     pickle.dump(tmp_dict, f, pickle.HIGHEST_PROTOCOL)
with open('./datasets/traj_debug/traj_dict_626.pkl', 'wb') as f:
    pickle.dump(tmp_dict, f, pickle.HIGHEST_PROTOCOL)



