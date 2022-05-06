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
import multiprocessing
from scipy.ndimage.filters import median_filter


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

    # post-process qpos
    # set the feet on ground plane
    qpos_traj[:, 2] += args.offset_z

    time_cost_get_poses = time.time() - time0_get_poses  # time spend.
    print('-> get_poses spends {:.2f}s on {} with {:0>6d} frames'.format(time_cost_get_poses, bvh_file, poses.shape[0]))
    return qpos_traj

def bvh2traj(file):
    print('start extracting trajectory from %s' % file)
    qpos_traj = get_poses(file)
    name = os.path.splitext(os.path.basename(file))[0]
    # bvh_dir = os.path.dirname(file)
    # traj_p_folder = bvh_dir.replace('traj_bvh', 'traj_p')
    traj_file = '%s/datasets/traj_p/%s_traj.p' % (args.mocap_folder, name)
    pickle.dump(qpos_traj, open(traj_file, 'wb'))
    print('save trajectory to %s' % traj_file)


if __name__=='__main__':
    """
    bvh to traj.p to expert
    python ./pose_imitation/data_process/convert_clip_multiprocess.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--model-id', type=str, default='humanoid_h36m_v4')
    parser.add_argument('--mocap-folder', type=str, default='debug')
    parser.add_argument('--mocap-fr', type=int, default=30)
    parser.add_argument('--dt', type=float, default=1 / 30)
    parser.add_argument('--offset-z', type=float, default=0.06)
    parser.add_argument('--num-threads', type=int, default=32)
    args = parser.parse_args()

    timer = Timer()
    traj_p_folder = '%s/datasets/traj_p' % args.mocap_folder
    recreate_dirs(traj_p_folder)

    model_file = 'assets/mujoco_models/%s.xml' % args.model_id
    model = load_model_from_path(model_file)
    body_qposaddr = get_body_qposaddr(model)
    timer.update_time('complete load XML')

    bvh_files = glob.glob(os.path.expanduser('%s/datasets/traj_bvh/*.bvh' % args.mocap_folder))
    bvh_files.sort()
    # if args.range is not None:
    #     bvh_files = bvh_files[args.range[0]: args.range[1]]
    print('bvh_files:', bvh_files)

    # init skeleton class.
    skt_bvh = bvh_files[0]
    exclude_bones = {'Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'End', 'Toe'}
    spec_channels = {'LeftForeArm': ['Zrotation'], 'RightForeArm': ['Zrotation'],
                     'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation']}
    skeleton = Skeleton()
    skeleton.load_from_bvh(skt_bvh, exclude_bones, spec_channels)
    # skeleton.write_xml('assets/mujoco_models/test.xml', 'assets/mujoco_models/template/humanoid_template.xml')

    # start
    task_lst = bvh_files
    num_threads = args.num_threads

    for ep in range(math.ceil(len(task_lst) / num_threads)):

        p_lst = []
        for i in range(num_threads):
            idx = ep * num_threads + i
            if idx >= len(task_lst):
                break
            p = multiprocessing.Process(target=bvh2traj, args=(task_lst[idx],))
            p_lst.append(p)

        for p in p_lst:
            p.start()

        for p in p_lst:
            p.join()

        print('complete ep:', ep)
    # end.
    timer.update_time('complete multiprocessing')

    # # save a traj_dict file,
    traj_dict_folder = '%s/datasets/traj_dict' % args.mocap_folder
    traj_dict_path = '{}/traj_dict.pkl'.format(traj_dict_folder)
    if os.path.exists(traj_dict_path):
        traj_dict = np.load(traj_dict_path, allow_pickle=True)
    else:
        recreate_dirs(traj_dict_folder)
        traj_dict = {}

    take_list = glob.glob('{}/*traj.p'.format(traj_p_folder))
    for take in take_list:
        take_name = take.split('/')[-1].split('_traj.')[0]
        orig_traj = np.load(take, allow_pickle=True)
        if not take_name in traj_dict:
            traj_dict[take_name] = {}
        traj_dict[take_name]['predicted_3d_qpos'] = orig_traj

    with open(traj_dict_path, 'wb') as f:
        pickle.dump(traj_dict, f, pickle.HIGHEST_PROTOCOL)





