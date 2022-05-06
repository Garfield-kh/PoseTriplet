import argparse
import os
import sys
import pickle
import multiprocessing
sys.path.append(os.getcwd())

from utils import *
from pose_imitation.envs.humanoid_v4 import HumanoidEnv
from pose_imitation.utils.statereg_dataset import Dataset
from pose_imitation.utils.posemimic_config import Config as EgoConfig


def get_expert(expert_qpos, lb, ub):
    """
    get qpos from intake file,
    assign qpos to simulator
    use env.sim.forward to get all information
    store as a expert
    """
    expert = {'qpos': expert_qpos}
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', 'head_pos', 'obs', 'ee_pos', 'ee_wpos', 'skt_pos', 'skt_wpos', 'sktvel', 'bquat', 'bangvel'}  # kh:加3D joint
    for key in feat_keys:
        expert[key] = []

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        # remove noisy hand data
        # qpos[slice(*env.body_qposaddr['LeftHand'])] = 0.0
        # qpos[slice(*env.body_qposaddr['RightHand'])] = 0.0
        env.data.qpos[:] = qpos
        env.sim.forward()
        rq_rmh = de_heading(qpos[3:7])
        obs = env.get_obs()
        ee_pos = env.get_ee_pos(env.cfg.obs_coord)
        ee_wpos = env.get_ee_pos(None)
        skt_pos = env.get_skeleton_pos(env.cfg.obs_coord)
        skt_wpos = env.get_skeleton_pos()
        bquat = env.get_body_quat()
        com = env.get_com()
        head_pos = env.get_body_com('Head').copy()
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd(prev_qpos, qpos, env.dt)
            rlinv = qvel[:3].copy()
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7], env.cfg.obs_coord)
            rangv = qvel[3:6].copy()  # # angular velocity is in root coord
            expert['qvel'].append(qvel)
            expert['rlinv'].append(rlinv)
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        expert['obs'].append(obs)
        expert['ee_pos'].append(ee_pos)
        expert['ee_wpos'].append(ee_wpos)
        expert['skt_pos'].append(skt_pos)
        expert['skt_wpos'].append(skt_wpos)
        expert['bquat'].append(bquat)
        expert['com'].append(com)
        expert['head_pos'].append(head_pos)
        expert['rq_rmh'].append(rq_rmh)
    expert['qvel'].insert(0, expert['qvel'][0].copy())
    expert['rlinv'].insert(0, expert['rlinv'][0].copy())
    expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
    expert['rangv'].insert(0, expert['rangv'][0].copy())

    # get expert body quaternions:
    for i in range(expert_qpos.shape[0]):
        if i > 0:
            bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
            expert['bangvel'].append(bangvel)
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    # get expert body skt_pos:
    for i in range(expert_qpos.shape[0]):
        if i > 0:
            sktvel = get_sktvel_fd(expert['skt_pos'][i - 1], expert['skt_pos'][i], env.dt)
            expert['sktvel'].append(sktvel)
    expert['sktvel'].insert(0, expert['sktvel'][0].copy())

    expert['qpos'] = expert['qpos'][lb:ub, :]
    for key in feat_keys:
        expert[key] = np.vstack(expert[key][lb:ub])
    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pos'][:, 2].min()

    return expert

def traj2expert(q, dataset, take_id):
    time0_get_expert = time.time()  # time start load.

    take = dataset.takes[take_id]
    _, lb, ub = dataset.msync[take]
    expert_qpos = dataset.orig_trajs[take_id]
    expert = get_expert(expert_qpos, lb, ub)

    q.put((take, expert))    #queue

    time_cost_get_expert = time.time() - time0_get_expert # time spend.
    print('-> get_expert spends {:.2f}s on ID{}:{} with {:0>6d} frames'.format(time_cost_get_expert,
                                                                          take_id, take, expert_qpos.shape[0]))


if __name__=='__main__':
    """
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-id', default='meta_subject_h36m')
    parser.add_argument('--model-id', type=str, default='humanoid_h36m_v4')  # XML
    parser.add_argument('--vis-model', default='humanoid_h36m_v4')
    # parser.add_argument('--out-id', default='subject_02')
    parser.add_argument('--of-folder', default='pose2d')  # 2d pose, optical-flow, not in use
    # add by kh for imitation task.
    parser.add_argument('--mocap-folder', type=str, default='debug')
    parser.add_argument('--num-threads', type=int, default=32)

    args = parser.parse_args()

    timer = Timer()
    traj_expert_folder = '%s/datasets/traj_expert' % args.mocap_folder
    recreate_dirs(traj_expert_folder)

    # prepare env, dataset.
    cfg_dict = {
        'meta_id': args.meta_id,
        'mujoco_model': args.model_id,
        'vis_model': args.vis_model,
        'obs_coord': 'heading',
    }
    cfg = EgoConfig(None, create_dirs=False, cfg_dict=cfg_dict, mocap_folder=args.mocap_folder)
    env = HumanoidEnv(cfg)
    dataset = Dataset(args.meta_id, 'all', 0, 'iter', False, 0, args=args, mocap_folder=args.mocap_folder)

    # start
    task_lst = dataset.takes
    num_threads = args.num_threads
    q = multiprocessing.Queue()

    num_sample = 0
    expert_dict = {}

    for ep in range(math.ceil(len(task_lst) / num_threads)):

        p_lst = []
        for i in range(num_threads):
            idx = ep * num_threads + i
            if idx >= len(task_lst):
                break
            p = multiprocessing.Process(target=traj2expert, args=(q, dataset, idx,))
            p_lst.append(p)

        for p in p_lst:
            p.start()

        for p in p_lst:
            take, expert = q.get()
            expert_dict[take] = expert

        for p in p_lst:
            p.join()

        print('complete ep:', ep)
    # end.
    timer.update_time('complete multiprocessing')

    # static:
    for take in expert_dict:
        expert = expert_dict[take]
        num_sample += expert['len']
        print(take, expert['len'], expert['qvel'].min(), expert['qvel'].max(), expert['head_height_lb'])

    print('meta: %s, total sample: %d, dataset length: %d' % (args.meta_id, num_sample, dataset.len))
    timer.update_time('complete gen expert')

    traj_expert_path = '%s/datasets/traj_expert/traj_expert.pkl' % args.mocap_folder
    pickle.dump(expert_dict, open(traj_expert_path, 'wb'))


