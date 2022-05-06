import argparse
import os
import sys
import pickle
import time
sys.path.append(os.getcwd())

from utils import *
from core.policy_gaussian import PolicyGaussian
from core.critic import Value
from models.mlp import MLP
from models.video_state_net import VideoStateNet
from pose_imitation.envs.humanoid_v4 import HumanoidEnv
from pose_imitation.core.agent_mimic import AgentEgo
from pose_imitation.utils.posemimic_config import Config
from pose_imitation.core.reward_function import reward_func, reward_name_list


# config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='subject_init')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--gpu-index', type=int, default=1)
parser.add_argument('--iter', type=int, default=0)  # for pretrain case
parser.add_argument('--show-noise', action='store_true', default=False)
parser.add_argument('--mocap-folder', type=str, default='checkpoint/exp_init/helix_0')

args = parser.parse_args()
if args.render:
    args.num_threads = 1
cfg = Config(args.cfg, create_dirs=not (args.render or args.iter > 0), mocap_folder=args.mocap_folder)

# log.
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
tb_logger = Logger(cfg.tb_dir) if not args.render else None
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'), file_handle=not args.render)

"""environment"""
env = HumanoidEnv(cfg)
env.seed(cfg.seed)
# if cfg.fix_head_lb:
#     env.set_fix_head_lb(cfg.fix_head_lb)
if cfg.set_fix_start_state:
    start_state = np.zeros(117)
    start_state[:7] = np.array([0,0,1,0,0,0,1])
    env.set_fix_sampling(start_state=start_state)
env.load_experts(cfg.takes['train'], cfg.expert_feat_file, cfg.cnn_feat_key)
cnn_feat_dim = env.cnn_feat[0].shape[-1]
actuators = env.model.actuator_names
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)

"""define actor and critic: vs: video state for better information fusion"""
policy_vs_net = VideoStateNet(cnn_feat_dim, cfg.policy_v_hdim, cfg.fr_margin, cfg.policy_v_net, cfg.policy_v_net_param, cfg.causal)
value_vs_net = VideoStateNet(cnn_feat_dim, cfg.value_v_hdim, cfg.fr_margin, cfg.value_v_net, cfg.value_v_net_param, cfg.causal)
policy_net = PolicyGaussian(MLP(state_dim + cfg.policy_v_hdim, cfg.policy_hsize, cfg.policy_htype), action_dim,
                            log_std=cfg.log_std, fix_std=cfg.fix_std)
value_net = Value(MLP(state_dim + cfg.value_v_hdim, cfg.value_hsize, cfg.value_htype))
if args.iter > 0:
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    policy_net.load_state_dict(model_cp['policy_dict'])
    policy_vs_net.load_state_dict(model_cp['policy_vs_dict'])
    value_net.load_state_dict(model_cp['value_dict'])
    value_vs_net.load_state_dict(model_cp['value_vs_dict'])
    running_state = model_cp['running_state']
to_device(device, policy_net, value_net, policy_vs_net, value_vs_net)

policy_params = list(policy_net.parameters()) + list(policy_vs_net.parameters())
value_params = list(value_net.parameters()) + list(value_vs_net.parameters())
if cfg.policy_optimizer == 'Adam':
    optimizer_policy = torch.optim.Adam(policy_params, lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
else:
    optimizer_policy = torch.optim.SGD(policy_params, lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
if cfg.value_optimizer == 'Adam':
    optimizer_value = torch.optim.Adam(value_params, lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
else:
    optimizer_value = torch.optim.SGD(value_params, lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)

# reward functions
expert_reward = reward_func[cfg.reward_id]
reward_name_list = reward_name_list[cfg.reward_id]

"""create agent"""
agent = AgentEgo(env=env, dtype=dtype, device=device, running_state=running_state,
                 custom_reward=expert_reward, mean_action=args.render and not args.show_noise,
                 render=args.render, num_threads=args.num_threads,
                 policy_net=policy_net, policy_vs_net=policy_vs_net,
                 value_net=value_net, value_vs_net=value_vs_net,
                 optimizer_policy=optimizer_policy, optimizer_value=optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                 gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                 policy_grad_clip=[(policy_params, 40)])


def pre_iter_update(i_iter):
    cfg.update_adaptive_params(i_iter)
    agent.set_noise_rate(cfg.adp_noise_rate)
    set_optimizer_lr(optimizer_policy, cfg.adp_policy_lr)
    if cfg.fix_std:
        policy_net.action_log_std.fill_(cfg.adp_log_std)
    return


def main_loop():

    if args.render:
        pre_iter_update(args.iter)
        agent.sample(1e8)
    else:
        for i_iter in range(args.iter, cfg.max_iter_num):
            """generate multiple trajectories that reach the minimum batch_size"""
            pre_iter_update(i_iter)
            batch, log = agent.sample(cfg.min_batch_size)
            agent.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)    # set end reward

            """update networks"""
            t0 = time.time()
            agent.update_params(batch)
            t1 = time.time()

            """logging"""
            c_info = log.avg_c_info
            logger.info(
                '{}\tT_sample {:.2f}\tT_update {:.2f}\tR_avg {:.4f} {}'
                '\tR_range ({:.4f}, {:.4f})\teps_len_avg {:.2f}'
                .format(i_iter, log.sample_time, t1 - t0, log.avg_c_reward,
                        np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=','),
                        log.min_c_reward, log.max_c_reward, log.avg_episode_reward))

            tb_logger.scalar_summary('total_reward', log.avg_c_reward, i_iter)
            tb_logger.scalar_summary('episode_len', log.avg_episode_reward, i_iter)
            for i in range(c_info.shape[0]):
                # tb_logger.scalar_summary('reward_%s' % reward_name_list[i], c_info[i], i_iter)
                tb_logger.scalar_summary('reward_detail/%s' % reward_name_list[i], c_info[i], i_iter)

            if cfg.save_model_interval > 0 and (i_iter+1) % cfg.save_model_interval == 0:
                with to_cpu(policy_net, value_net, policy_vs_net, value_vs_net):
                    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_iter + 1)
                    model_cp = {'policy_dict': policy_net.state_dict(), 'policy_vs_dict': policy_vs_net.state_dict(),
                                'value_dict': value_net.state_dict(), 'value_vs_dict': value_vs_net.state_dict(),
                                'running_state': running_state}
                    pickle.dump(model_cp, open(cp_path, 'wb'))

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        logger.info('training done!')


main_loop()
