import argparse
import os
import sys
import pickle
import math
import time
import numpy as np
sys.path.append(os.getcwd())
import subprocess

from pose_imitation.utils.metrics import *
from pose_imitation.utils.tools import *
from pose_imitation.utils.posemimic_config import Config
# from pose_imitation.envs.humanoid_v1 import HumanoidEnv
from envs.visual.humanoid_vis import HumanoidVisEnv

"""
screen shot video
"""

parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_h36m_v4')  # humanoid_1205_vis_double_v1
# parser.add_argument('--multi-vis-model', default='humanoid_h36m_v4_vis')
parser.add_argument('--vis-model-num', default='single')  # single, double, multi
# parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--posemimic-cfg', default='subject_init')
# parser.add_argument('--statereg-cfg', default='subject_02')
# parser.add_argument('--posemimic-iter', type=int, default=3000)
# parser.add_argument('--statereg-iter', type=int, default=100)
# parser.add_argument('--posemimic-tag', default='')
parser.add_argument('--traj-dict-path', default='./checkpoint/exp_init/helix_0/datasets/traj_dict/traj_dict.pkl')
parser.add_argument('--mocap-folder', type=str, default='debug')  #
parser.add_argument('--qpos-key', default='predicted_3d_qpos')  #
parser.add_argument('--eqpos-key', default='predicted_3d_qpos')  #
parser.add_argument('--data', default='train')
parser.add_argument('--mode', default='vis')
parser.add_argument('--take_ind_list', default=None)
args = parser.parse_args()


cfg = Config(args.posemimic_cfg)
dt = 1 / 30.0

"""
algo_res['traj_pred'][take][args.qpos_key]
em_res{
'traj_pred':{'take_name': tx59}
'traj_orig':{'take_name': tx59}
}
"""
if not args.mocap_folder == 'debug':
    traj_dict_path = '{}/datasets/traj_dict/traj_dict.pkl'.format(args.mocap_folder)
else:
    traj_dict_path = args.traj_dict_path
traj_dict = np.load(traj_dict_path, allow_pickle=True)
em_res = {
    'traj_pred': traj_dict,
    'traj_orig': traj_dict,
}
sr_res = {
    'traj_pred': traj_dict,
    'traj_orig': traj_dict,
}


if args.mode == 'vis':
    """visualization"""

    def key_callback(key, action, mods):
        global T, fr, paused, stop, reverse, algo_ind, take_ind, ss_ind, show_gt, mfr_int

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            T *= 1.5
        elif key == glfw.KEY_F:
            T = max(1, T / 1.5)
        elif key == glfw.KEY_R:
            stop = True
        elif key == glfw.KEY_W:
            fr = 0
            update_pose()
        elif key == glfw.KEY_S:
            reverse = not reverse
        elif key == glfw.KEY_Z:
            fr = 0
            take_ind = (take_ind - 1) % len(takes)
            load_take()
            update_pose()
        elif key == glfw.KEY_C:
            fr = 0
            take_ind = (take_ind + 1) % len(takes)
            load_take()
            update_pose()
        # elif key == glfw.KEY_X:
        #     save_screen_shots(env_vis.viewer.window, 'out/%04d.png' % ss_ind)
        #     ss_ind += 1
        elif key == glfw.KEY_X:
            save_video()
        elif glfw.KEY_1 <= key < glfw.KEY_1 + len(algos):
            algo_ind = key - glfw.KEY_1
            load_take()
            update_pose()
        elif key == glfw.KEY_0:
            show_gt = not show_gt
            update_pose()
        elif key == glfw.KEY_MINUS:
            mfr_int -= 1
            update_pose()
        elif key == glfw.KEY_EQUAL:
            mfr_int += 1
            update_pose()
        elif key == glfw.KEY_RIGHT:
            if fr < traj_orig.shape[0] - 1:
                fr += 1
            update_pose()
        elif key == glfw.KEY_LEFT:
            if fr > 0:
                fr -= 1
            update_pose()
        elif key == glfw.KEY_SPACE:
            paused = not paused
        else:
            return False

        return True

    def update_pose():
        print('take_ind: %d, fr: %d, mfr int: %d' % (take_ind, fr, mfr_int))
        if args.vis_model_num == 'multi':
            nq = 59
            traj = traj_orig if show_gt else traj_pred
            num_model = env_vis.model.nq // nq
            hq = get_heading_q(traj_orig[fr, 3:7])
            rel_q = quaternion_multiply(hq, quaternion_inverse(get_heading_q(traj[fr, 3:7])))
            vec = quat_mul_vec(hq, np.array([0, -1, 0]))[:2]
            for i in range(num_model):
                fr_m = min(fr + i * mfr_int, traj.shape[0] - 1)
                env_vis.data.qpos[i*nq: (i+1)*nq] = traj[fr_m, :]
                env_vis.data.qpos[i*nq + 3: i*nq + 7] = quaternion_multiply(rel_q, traj[fr_m, 3:7])
                env_vis.data.qpos[i*nq: i*nq + 2] = traj_orig[fr, :2] + vec * 0.8 * i
        elif args.vis_model_num == 'single':
            env_vis.data.qpos[:] = traj_pred[fr, :]
        elif args.vis_model_num == 'double':
            nq = env_vis.model.nq // 2
            env_vis.data.qpos[:nq] = traj_pred[fr, :]
            env_vis.data.qpos[nq:] = traj_orig[fr, :]
            # add x offset
            env_vis.data.qpos[nq] += 1.0
        else:
            assert False, 'invalid vis_model_num: {}'.format(args.vis_model_num)
        env_vis.sim_forward()

    def load_take():
        global traj_pred, traj_orig
        algo_res, algo = algos[algo_ind]  # res: rlt
        if algo_res is None:
            return
        take = takes[take_ind]
        print('%s ---------- %s' % (algo, take))
        traj_pred = algo_res['traj_pred'][take][args.qpos_key]
        traj_orig = algo_res['traj_orig'][take][args.eqpos_key]

    def save_video():
        global fr
        # save video
        # num_fr = traj_pred.shape[0]
        num_fr = min(300, traj_pred.shape[0])

        args.preview = False
        # args.video_dir = 'out'
        # args.video_dir = args.traj_dict_path[:-2] + '_data-' + args.data
        # args.video_dir = '{}/take-{}'.format(args.traj_dict_path[:-2], takes[take_ind])
        args.video_dir = '{}'.format(args.traj_dict_path[:-2])

        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(num_fr):
            fr = fr
            update_pose()
            for _ in range(20):  # render more times
                env_vis.render()
            if not args.preview:
                t0 = time.time()
                save_screen_shots(env_vis.viewer.window, f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, num_fr, time.time() - t0))

        if not args.preview:
            out_name = '{}/{}-{}-{}.mp4'.format(args.traj_dict_path[:-2], args.data, take_ind, takes[take_ind])
            cmd = ['ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)

    def set_viewer():
        env_vis.viewer._hide_overlay = True  # hide the entire overlay.
        # env_vis.viewer.cam.distance = env_vis.model.stat.extent * 1.5
        # env_vis.viewer.cam.elevation = -10
        # env_vis.viewer.cam.azimuth = 90


    traj_pred = None
    traj_orig = None
    vis_model_file = 'assets/mujoco_models/%s.xml' % (args.vis_model)
    env_vis = HumanoidVisEnv(vis_model_file, 1, focus=not args.vis_model_num == 'multi')
    env_vis.set_custom_key_callback(key_callback)
    takes = cfg.takes[args.data]
    algos = [(em_res, 'ego mimic'), (sr_res, 'state reg')]
    algo_ind = 0
    take_ind = 0
    ss_ind = 0  # screen shot
    mfr_int = 10
    show_gt = False
    load_take()

    """render or select part of the clip"""
    T = 10
    fr = 0
    paused = False
    stop = False
    reverse = False

    if args.take_ind_list:
        set_viewer()
        # for take_ind in [int(x) for x in args.take_ind_list.split(',')]:
        #     load_take()
        #     save_video()
        # for take_ind in np.arange(0, 600, 4):
        for take_ind in np.arange(0, 22, 1):
            load_take()
            save_video()

    else:
        # show video
        update_pose()
        t = 0
        while not stop:
            if t >= math.floor(T):
                if not reverse and fr < traj_orig.shape[0] - 1:
                    fr += 1
                    update_pose()
                elif reverse and fr > 0:
                    fr -= 1
                    update_pose()
                t = 0

            env_vis.render()
            if not paused:
                t += 1







