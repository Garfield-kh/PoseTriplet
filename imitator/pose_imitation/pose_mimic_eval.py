import argparse
import os
import sys
import pickle
import time
sys.path.append(os.getcwd())
import multiprocessing

from utils import *
from core.policy_gaussian import PolicyGaussian
from core.critic import Value
from models.mlp import MLP
from models.video_state_net import VideoStateNet
from models.video_reg_net import VideoRegNet
from envs.visual.humanoid_vis import HumanoidVisEnv
from pose_imitation.envs.humanoid_v4 import HumanoidEnv
from pose_imitation.utils.posemimic_config import Config
from pose_imitation.core.reward_function import reward_func


# config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='subject_03')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--save-gif', action='store_true', default=False)  # plot sktwpos in gif
parser.add_argument('--gif-ds', type=int, default='5')
parser.add_argument('--iter', type=int, default='3000')
parser.add_argument('--expert-ind', type=int, default=-1)  # take_ind
parser.add_argument('--sync', action='store_true', default=False)
parser.add_argument('--causal', action='store_true', default=False)
parser.add_argument('--data', default='test')
parser.add_argument('--show-noise', action='store_true', default=False)
parser.add_argument('--fail-safe', default='naivefs') # 'valuefs'
parser.add_argument('--valuefs-factor', type=float, default='0.6')
parser.add_argument('--mocap-folder', type=str, default='debug')
parser.add_argument('--num-threads', type=int, default=32)

args = parser.parse_args()
cfg = Config(args.cfg, create_dirs=False, mocap_folder=args.mocap_folder)

# 准备log文件.
dtype = torch.float64
torch.set_default_dtype(dtype)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

"""environment"""
env = HumanoidEnv(cfg)
env.load_experts(cfg.takes[args.data], cfg.expert_feat_file, cfg.cnn_feat_key)
env_vis = HumanoidVisEnv(cfg.vis_model_file, 10)
env.seed(cfg.seed)
epos = None
cnn_feat_dim = env.cnn_feat[0].shape[-1]
actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
body_qposaddr = get_body_qposaddr(env.model)
# if args.fail_safe == 'naivefs':
#     env.set_fix_head_lb(cfg.fix_head_lb)
if cfg.set_fix_start_state:
    start_state = np.zeros(117)
    start_state[:7] = np.array([0,0,1,0,0,0,1])
    env.set_fix_sampling(start_state=start_state)

"""load policy net"""
policy_vs_net = VideoStateNet(cnn_feat_dim, cfg.policy_v_hdim, cfg.fr_margin, cfg.policy_v_net, cfg.policy_v_net_param, cfg.causal)
value_vs_net = VideoStateNet(cnn_feat_dim, cfg.value_v_hdim, cfg.fr_margin, cfg.value_v_net, cfg.value_v_net_param, cfg.causal)
policy_net = PolicyGaussian(MLP(state_dim + cfg.policy_v_hdim, cfg.policy_hsize, cfg.policy_htype), action_dim,
                            log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim + cfg.value_v_hdim, cfg.value_hsize, cfg.value_htype))
cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
logger.info('loading policy net from checkpoint: %s' % cp_path)
model_cp = pickle.load(open(cp_path, "rb"))
policy_net.load_state_dict(model_cp['policy_dict'])
policy_vs_net.load_state_dict(model_cp['policy_vs_dict'])
value_net.load_state_dict(model_cp['value_dict'])
value_vs_net.load_state_dict(model_cp['value_vs_dict'])
running_state = model_cp['running_state']
value_stat = RunningStat(1)

# count_param
count_param(logger, policy_vs_net, 'policy_vs_net')
count_param(logger, value_vs_net, 'value_vs_net')
count_param(logger, policy_net, 'policy_net')
count_param(logger, value_net, 'value_net')

"""load state net"""

# to_test(policy_vs_net, policy_net, value_vs_net, value_net, state_net)
to_test(policy_vs_net, policy_net, value_vs_net, value_net)

# reward functions
expert_reward = reward_func[cfg.reward_id]


def render():
    env_vis.data.qpos[:env.model.nq] = env.data.qpos.copy()
    env_vis.data.qpos[env.model.nq:] = epos
    env_vis.data.qpos[env.model.nq] += 1.0
    env_vis.sim_forward()
    env_vis.render()


def reset_env_state(state, ref_qpos):
    qpos = ref_qpos.copy()
    qpos[2:] = state[:qpos.size - 2]
    qvel = state[qpos.size - 2:]
    align_human_state(qpos, qvel, ref_qpos)
    env.set_state(qpos, qvel)
    return env.get_obs()


def eval_expert(expert_ind):
    global epos

    expert_name = env.expert_list[expert_ind]
    logger.info('Testing on expert trajectory %s' % expert_name)

    traj_pred = []
    traj_orig = []
    vel_pred = []
    skt_wpos_orig = []
    skt_wpos_pred = []
    num_reset = 0
    log_reset = []
    reward_episode = 0
    data_len = env.cnn_feat[expert_ind].shape[0]
    test_len = data_len - 2 * cfg.fr_margin
    env.set_fix_sampling(expert_ind, cfg.fr_margin, test_len)

    state = env.reset()
    cnn_feat = tensor(env.get_episode_cnn_feat())
    policy_vs_net.initialize(cnn_feat)
    value_vs_net.initialize(cnn_feat)


    state_pred = env.get_episode_export_qpos_qvel()
    if cfg.set_fix_start_state:
        state = reset_env_state(start_state[2:], env.data.qpos)
    else:
        state = reset_env_state(state_pred[0, :], env.data.qpos)
    if running_state is not None:
        state = running_state(state, update=False)

    for t in range(test_len):

        ind = env.get_expert_index(t)
        epos = env.get_expert_attr('qpos', ind).copy()
        skt_wpos_e = env.get_expert_attr('skt_wpos', ind).copy()
        skt_wpos = env.get_skeleton_pos().copy()
        traj_pred.append(env.data.qpos.copy())
        traj_orig.append(epos.copy())
        vel_pred.append(env.data.qvel.copy())
        skt_wpos_orig.append(skt_wpos_e)
        skt_wpos_pred.append(skt_wpos)

        if args.sync:
            epos[:3] = quat_mul_vec(env.expert['rel_heading'], epos[:3] - env.expert['start_pos']) + env.expert['sim_pos']
            epos[3:7] = quaternion_multiply(env.expert['rel_heading'], epos[3:7])

        if args.render:
            render()

        if args.causal:
            policy_vs_net.initialize(cnn_feat[:t + 2*cfg.fr_margin + 1])
            policy_vs_net.t = t

        """learner policy"""
        state_var = tensor(state, dtype=dtype).unsqueeze(0)
        policy_vs_out = policy_vs_net(state_var)
        value_vs_out = value_vs_net(state_var)
        value = value_net(value_vs_out).item()
        value_stat.push(np.array([value]))

        action = policy_net.select_action(policy_vs_out, mean_action=not args.show_noise)[0].numpy()
        next_state, reward, done, info = env.step(action)
        if running_state is not None:
            next_state = running_state(next_state, update=False)

        reward, cinfo = reward_func[cfg.reward_id](env, state, action, info)
        logger.debug("{} {:.2f} {} {:.2f}".format(t, reward, np.array2string(cinfo, formatter={'all': lambda x: '%.4f' % x}), value))

        reward_episode += reward

        if info['end']:
            break

        if args.fail_safe == 'valuefs' and value < args.valuefs_factor * value_stat.mean[0] or \
                args.fail_safe == 'naivefs' and info['fail']:
            logger.info('reset state!')
            num_reset += 1
            log_reset.append(t)
            if cfg.set_fix_start_state:
                state = reset_env_state(start_state[2:], env.data.qpos)
            else:
                state = reset_env_state(state_pred[t+1, :], env.data.qpos)
            if running_state is not None:
                state = running_state(state, update=False)
        else:
            state = next_state

    return np.vstack(traj_pred), np.vstack(traj_orig), np.vstack(vel_pred), np.vstack(skt_wpos_pred), np.vstack(skt_wpos_orig), num_reset, log_reset


# if expert_ind is defined, then keeping visualizing this trajectory
if args.expert_ind >= 0:
    for i in range(100):
        eval_expert(args.expert_ind)
elif args.render or args.save_gif:
    # traj_pred = {}
    # traj_orig = {}
    # vel_pred = {}
    # skt_wpos_pred = {}
    result_dict = {}
    num_reset = 0
    for i, take in enumerate(env.expert_list):
        # traj_pred[take], traj_orig[take], vel_pred[take], skt_wpos_pred[take], t_num_reset = eval_expert(i)
        traj_pred, traj_orig, vel_pred, skt_wpos_pred, skt_wpos_orig, t_num_reset, log_reset = eval_expert(i)
        # result_dict[take] = {'skt_wpos': skt_wpos_pred, 'skt_wpos_orig': skt_wpos_orig,}
        result_dict[take] = {'traj_pred': traj_pred, 'traj_orig': traj_orig,
                             'skt_wpos': skt_wpos_pred, 'skt_wpos_orig': skt_wpos_orig,
                             't_num_reset': t_num_reset, 'log_reset': log_reset,
                             }
        num_reset += t_num_reset
    # results = {'traj_pred': traj_pred, 'traj_orig': traj_orig, 'vel_pred': vel_pred, 'skt_wpos': skt_wpos_pred}
    meta = {'algo': 'ego_mimic', 'num_reset': num_reset}
    fs_tag = '' if args.fail_safe == 'valuefs' else '_' + args.fail_safe
    c_tag = '_causal' if args.causal else ''
    res_path = '%s/iter_%04d_%s%s%s.p' % (cfg.result_dir, args.iter, args.data, fs_tag, c_tag)
    # pickle.dump((results, meta), open(res_path, 'wb'))
    pickle.dump(result_dict, open(res_path, 'wb'))
    logger.info('num reset: %d' % num_reset)
    logger.info('saved results to %s' % res_path)

    if args.save_gif:
        from tmp.viz import save_3dpose_gif
        for i, take in enumerate(env.expert_list):
            tmp_skt_wpos_pred = result_dict[take]['skt_wpos'].reshape(-1, 16, 3)
            tmp_skt_wpos_orig = result_dict[take]['skt_wpos_orig'].reshape(-1, 16, 3)
            save_name = '%s/iter_%04d_%s%s%s/%s.gif' % (cfg.result_dir, args.iter, args.data, fs_tag, c_tag, take)
            mkd(save_name)
            save_3dpose_gif(tmp_skt_wpos_pred, tmp_skt_wpos_orig, save_name, args.gif_ds)
            logger.info('saved results gif to %s' % save_name)

else:
    def get_eval_expert(q, expert_ind):
        time0_get_expert = time.time()  # time start load.

        take = env.expert_list[expert_ind]
        traj_pred, traj_orig, vel_pred, skt_wpos_pred, skt_wpos_orig, t_num_reset, log_reset = eval_expert(expert_ind)
        # tmp_results = {'traj_pred': traj_pred, 'traj_orig': traj_orig, 'vel_pred': vel_pred,
        #                  'skt_wpos': skt_wpos_pred, 't_num_reset':t_num_reset}
        tmp_results = {'traj_pred': traj_pred, 'traj_orig': traj_orig,
                       'skt_wpos': skt_wpos_pred, 'skt_wpos_orig': skt_wpos_orig,
                       't_num_reset':t_num_reset, 'log_reset': log_reset,}
        # tmp_results = {'skt_wpos': skt_wpos_pred, 't_num_reset':t_num_reset}

        q.put((take, tmp_results))  # queue

        time_cost_get_expert = time.time() - time0_get_expert  # time spend.
        print('-> get_eval_expert spends {:.2f}s on ID{}:{} with {:0>6d} frames'.format(time_cost_get_expert,
                                                                                   expert_ind, take, skt_wpos_pred.shape[0]))

    # start
    # task_lst = np.arange(0, len(env.expert_list), 1)
    task_lst = env.expert_list
    num_threads = args.num_threads
    q = multiprocessing.Queue()
    timer = Timer()

    # num_sample = 0
    result_dict = {}

    for ep in range(math.ceil(len(task_lst) / num_threads)):

        p_lst = []
        for i in range(num_threads):
            idx = ep * num_threads + i
            if idx >= len(task_lst):
                break
            p = multiprocessing.Process(target=get_eval_expert, args=(q, idx,))
            p_lst.append(p)

        for p in p_lst:
            p.start()

        for p in p_lst:
            take, tmp_results = q.get()
            result_dict[take] = tmp_results

        for p in p_lst:
            p.join()

        print('complete ep:', ep)
    # end.
    timer.update_time('complete multiprocessing')

    # static:
    num_reset = 0
    for take in result_dict:
        tmp_results = result_dict[take]
        num_reset += tmp_results['t_num_reset']

    fs_tag = '' if args.fail_safe == 'valuefs' else '_' + args.fail_safe
    c_tag = '_causal' if args.causal else ''
    res_path = '%s/iter_%04d_%s%s%s.p' % (cfg.result_dir, args.iter, args.data, fs_tag, c_tag)
    # pickle.dump((results, meta), open(res_path, 'wb'))
    pickle.dump(result_dict, open(res_path, 'wb'))
    logger.info('num reset: %d' % num_reset)
    logger.info('saved results to %s' % res_path)

