import numpy as np
from envs.common import mujoco_env
from gym import spaces
from utils import *
from utils.transformation import quaternion_from_euler, rotation_from_quaternion, quaternion_about_axis
from mujoco_py import functions as mjf
import pickle
import time
import cv2 as cv
from scipy.linalg import cho_solve, cho_factor

"""

"""


class HumanoidEnv(mujoco_env.MujocoEnv):

    def __init__(self, cfg):
        mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file, 15)
        self.cfg = cfg
        # visualization
        self.save_video = False
        self.video_res = (224, 224)
        self.video_dir = cfg.video_dir
        self.set_cam_first = set()
        self.subsample_rate = 1
        # env specific
        self.end_reward = 0.0
        self.start_ind = 0
        self.body_qposaddr = get_body_qposaddr(self.model)  # 一个带body_name:index_address的dict:
        # {'Hips': (0, 7), 'Spine': (7, 10), 'Spine1': (10, 13), 'Spine2': (13, 16), 'Spine3': (16, 19),
        # 'Neck': (19, 22), 'Head': (22, 25),
        # 'RightShoulder': (25, 28), 'RightArm': (28, 31), 'RightForeArm': (31, 32), 'RightHand': (32, 35),
        # 'LeftShoulder': (35, 38), 'LeftArm': (38, 41), 'LeftForeArm': (41, 42), 'LeftHand': (42, 45),
        # 'RightUpLeg': (45, 48), 'RightLeg': (48, 49), 'RightFoot': (49, 52),
        # 'LeftUpLeg': (52, 55), 'LeftLeg': (55, 56), 'LeftFoot': (56, 59)}
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.skt_pos = self.get_skeleton_pos(self.cfg.obs_coord)
        self.prev_skt_pos = None
        # expert
        self.expert_ind = None
        self.expert_id = None
        self.expert_list = None     # name only
        self.expert_arr = None      # store actual experts dict
        self.expert = None
        # self.cnn_feat = None
        self.qpos_target = None
        # fixed sampling
        self.fix_expert_ind = None
        self.fix_start_ind = None
        self.fix_len = None
        self.fix_start_state = None
        self.fix_cnn_feat = None
        self.fix_head_lb = None
        # set agent
        self.set_spaces()
        self.set_model_params()

    def load_experts(self, expert_list, expert_feat_file, cnn_feat_key='qpos'):  # 默认expert是qpos, kgail的时候用2dpose
        self.expert_ind = 0
        self.expert_list = expert_list
        expert_dict = pickle.load(open(expert_feat_file, 'rb'))
        self.expert_arr = [expert_dict[x] for x in self.expert_list]  # expert_arr 其实是一个dict, 包含qpos qvel啥的.
        self.set_expert(0)
        # cnn_feat_dict, _ = pickle.load(open(cnn_feat_file, 'rb'))
        # self.cnn_feat = [cnn_feat_dict[x] for x in self.expert_list]
        # self.cnn_feat = [expert_dict[x][expert_key] for x in self.expert_list]  # 从这里开始, cnn feat就是qpos了.
        # 这里设定一下, 可能需要输入的是3D的root traj和vel. 14个channel.
        if cnn_feat_key == 'root_traj':
            tmp_qpos =  [expert_dict[x]['qpos'][:, :7].astype('float64') for x in self.expert_list]
            tmp_qvel =  [expert_dict[x]['qvel'][:, :6].astype('float64') for x in self.expert_list]
            self.cnn_feat = [np.concatenate([item1, item2], axis=1) for item1, item2 in zip(tmp_qpos, tmp_qvel)]
        elif cnn_feat_key == 'qposvel':
            tmp_qpos =  [expert_dict[x]['qpos'].astype('float64') for x in self.expert_list]
            tmp_qvel =  [expert_dict[x]['qvel'].astype('float64') for x in self.expert_list]
            self.cnn_feat = [np.concatenate([item1, item2], axis=1) for item1, item2 in zip(tmp_qpos, tmp_qvel)]
        elif cnn_feat_key == 'masked_qposvel':
            tmp_qpos =  [expert_dict[x]['qpos'].astype('float64') for x in self.expert_list]
            tmp_qvel =  [expert_dict[x]['qvel'].astype('float64') for x in self.expert_list]
            self.cnn_feat = [np.concatenate([item1, item2], axis=1) for item1, item2 in zip(tmp_qpos, tmp_qvel)]
            self.cnn_feat = [mask_features(item, self.cfg.masked_freq) for item in self.cnn_feat]  # in train
            # self.cnn_feat = [mask_features_withshuffle(item, self.cfg.masked_freq) for item in self.cnn_feat]  # in test
        else:
            self.cnn_feat = [expert_dict[x][cnn_feat_key].reshape(expert_dict[x][cnn_feat_key].shape[0], -1)\
                                 .astype('float64') for x in self.expert_list]  # 从这里开始, cnn feat就是qpos/2dpose了.

    def set_model_params(self):
        if self.cfg.action_type == 'torque' and hasattr(self.cfg, 'j_stiff'):
            self.model.jnt_stiffness[1:] = self.cfg.j_stiff
            self.model.dof_damping[6:] = self.cfg.j_damp

    def set_spaces(self):
        """
        不同的地方, ego pose这里的action space上下限是bound直接拿的, 全是0, rfc的是+-1,
        可能和model的xml有关系.
        """
        # action_space
        # bounds = self.model.actuator_ctrlrange.copy()
        # self.action_space = spaces.Box(low=bounds[:, 0], high=bounds[:, 1], dtype=np.float32)
        cfg = self.cfg
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        self.vf_dim = 6 if cfg.residual_force else 0
        self.meta_dim = 15 * 2 if cfg.meta_control else 0
        self.action_dim = self.ndof + self.vf_dim + self.meta_dim
        self.action_space = spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim), dtype=np.float32)

        # observation_space
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def get_obs(self):
        obs = self.get_full_obs()
        return obs

    def get_full_obs(self):
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])  # qpos的前两个, 也就是 root的xy不计入考虑.
        # vel
        if self.cfg.obs_vel == 'root':
            obs.append(qvel[:6])
        elif self.cfg.obs_vel == 'full':
            obs.append(qvel)
        # phase  # 好像没有用到.
        if hasattr(self.cfg, 'obs_phase') and self.cfg.obs_phase:
            phase = min(self.cur_t / self.cfg.env_episode_len, 1.0)
            obs.append(np.array([phase]))
        # 可能需要把expert的next frame qpos直接塞到这里来, 加快训练速度, 以及丢掉vs网络.
        seq_len = 3
        if self.cfg.masked_freq is None:  # when use mask training, remove the target pose.
            if self.expert is not None:
                qposseq_target = self.get_poseseq_target(seq_len)
            else:
                print('use qpos as qpos target for init obs space, only once!')
                qposseq_target = np.ones(qpos.shape[0]*seq_len)   # None for init obs space

            obs.append(qposseq_target)
        # all in
        obs = np.concatenate(obs)
        return obs

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Head']
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_skeleton_pos(self, transform=None):
        '''
        get skeleton - kehong
        这里定义的16个点，可能和H36M/3DHP不太统一，可以考虑去掉躯干部分。
        _body_name2id: {'world': 0, 'Hips': 1, 'Spine': 2, 'Spine1': 3, 'Spine2': 4, 'Spine3': 5, 'Neck': 6, 'Head': 7,
        'RightShoulder': 8, 'RightArm': 9, 'RightForeArm': 10, 'RightHand': 11,
        'LeftShoulder': 12, 'LeftArm': 13, 'LeftForeArm': 14, 'LeftHand': 15,
        'RightUpLeg': 16, 'RightLeg': 17, 'RightFoot': 18,
        'LeftUpLeg': 19, 'LeftLeg': 20, 'LeftFoot': 21}
        '''
        data = self.data
        skt_name = [
            'Hips',
            'RightUpLeg', 'RightLeg', 'RightFoot',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot',
            # 'Spine2', 'Spine3', 'Neck', 'Head',
            'Spine2', 'Neck', 'Head',
            'LeftArm', 'LeftForeArm', 'LeftHand',
            'RightArm', 'RightForeArm', 'RightHand',
            ]
        # skt_name = self.body_qposaddr  # 尝试打印所有body_qposaddr的点出来.
        skt_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in skt_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            skt_pos.append(bone_vec)
        return np.concatenate(skt_pos)

    def get_body_quat(self):
        qpos = self.data.qpos.copy()
        body_quat = [qpos[3:7]]
        for body in self.model.body_names[1:]:
            if body == 'Hips':
                continue
            start, end = self.body_qposaddr[body]
            euler = np.zeros(3)
            euler[:end - start] = qpos[start:end]  # 这里是对胳膊和腿关节的特例, 让z保留.
            quat = quaternion_from_euler(euler[0], euler[1], euler[2], axes='rxyz')  # kh:和pose2bvh对应.
            body_quat.append(quat)
        body_quat = np.concatenate(body_quat)
        return body_quat

    def get_com(self):
        return self.data.subtree_com[0, :].copy()


    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        """
        ego的kp kd写在这个方程里面, 是一串list, rfc的从上一级拿,
        rfc的cho_factor多了一些flag, 需要查看一下.
        """
        dt = self.model.opt.timestep
        nv = self.model.nv
        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(self.model.nv, self.model.nv)
        C = self.data.qfrc_bias.copy()
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(cho_factor(M + K_d*dt, overwrite_a=True, check_finite=False),
                            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]), overwrite_b=True, check_finite=False)
        return q_accel.squeeze()


    def compute_torque(self, ctrl, i):
        """
        ego的直接是当前qpos - ctrl的结果,
        rfc的是把ctrl先scale, 然后加一个ref qpos,作为target pose, 去和当前qpos以及速度做误差.
        他们都把qvel当qvel_error
        需要看看这里的原理是啥.
        rfc额外参数: a_scale a_ref - 这两个在ego的do_simulation
        """
        cfg = self.cfg
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof] * cfg.a_scale
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # base_pos = cfg.a_ref
        base_pos = self.qpos_ref[7:] * 1.  # simpoe pose 这里是直接用的target qpos, 但是我们的target qpos噪音大.
        # base_pos = qpos[7:] * 1.   # 换成当前qpos试试. 不能用这个,pose坍塌了. 可以考虑把target pose换成上一个qpos, 但不是这儿.
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        if cfg.meta_control:
            meta_control = ctrl[-self.meta_dim:][i*2:(i+1)*2]
            k_p[6:] = cfg.jkp * np.abs(meta_control[0])
            k_d[6:] = cfg.jkd * np.abs(meta_control[1])
        else:
            k_p[6:] = cfg.jkp
            k_d[6:] = cfg.jkd
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:]*dt - target_pos))  # qpos 59, 但是actuator只管后52个.
        qvel_err = qvel
        # qpos_err, qvel_err = self.pdcontrolrefinement(qpos_err, qvel_err)  # add kh, 0925
        if cfg.arm_pdclip:
            qpos_err = self.pdcontrolclip(qpos_err)  # add kh, 0927
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -cfg.jkp * qpos_err[6:] - cfg.jkd * qvel_err[6:]
        return torque


    def pdcontrolrefinement(self, qpos_err_in, qvel_err_in):
        """
        0925: 由于DL生成的pose可能出现奇怪的手臂角度, 导致arm的180旋转,这个时候会有转大圈和小圈的区别,
        这里对大圈做一下限制.
        结果: NAN了. 可能不行.
        :return:
        """
        assert False, 'not working'
        pi = np.pi
        qpos_err_out = np.where(qpos_err_in < - pi, qpos_err_in + 2 * pi, qpos_err_in)
        qpos_err_out = np.where(qpos_err_in > + pi, qpos_err_out - 2 * pi, qpos_err_out)
        qvel_err_out = np.where(qpos_err_in < - pi, qvel_err_in * -1, qvel_err_in)
        qvel_err_out = np.where(qpos_err_in > + pi, qvel_err_out * -1, qvel_err_out)
        return qpos_err_out, qvel_err_out

    def pdcontrolclip(self, qpos_err_in):
        """
        0925: 由于DL生成的pose可能出现奇怪的手臂角度, 导致arm的180旋转,
        这里对qpos
        注意: 这里qpos_error是58维的, qpos是59维的. 所以 28~30变成 27~29
        :return:
        """
        # pi = np.pi
        qpos_err_out = qpos_err_in * 1.
        arm_x_limit = np.deg2rad(100)
        # arm_y_limit = np.deg2rad(50)
        # arm_z_limit = np.deg2rad(100)
        arm_xyz_limit =  np.deg2rad(0.5)  # default 2
        # qpos_err_out[27] = np.clip(qpos_err_out[27], -arm_x_limit, arm_x_limit)
        # qpos_err_out[28] = np.clip(qpos_err_out[28], -arm_y_limit, arm_y_limit)
        # qpos_err_out[29] = np.clip(qpos_err_out[29], -arm_z_limit, arm_z_limit)
        # qpos_err_out[37] = np.clip(qpos_err_out[37], -arm_x_limit, arm_x_limit)
        # qpos_err_out[38] = np.clip(qpos_err_out[38], -arm_y_limit, arm_y_limit)
        # qpos_err_out[39] = np.clip(qpos_err_out[39], -arm_z_limit, arm_z_limit)
        if np.abs(qpos_err_out[27]) > arm_x_limit:
            qpos_err_out[27:30] = np.clip(qpos_err_out[27:30], -arm_xyz_limit, arm_xyz_limit)
        if np.abs(qpos_err_out[37]) > arm_x_limit:
            qpos_err_out[37:40] = np.clip(qpos_err_out[37:40], -arm_xyz_limit, arm_xyz_limit)

        return qpos_err_out



    """ RFC-Implicit """
    def rfc_implicit(self, vf):
        vf *= self.cfg.residual_force_scale
        hq = get_heading_q(self.data.qpos[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        self.data.qfrc_applied[:vf.shape[0]] = vf


    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cfg
        for i in range(n_frames):
            ctrl = action * 1.
            if cfg.action_type == 'position':
                torque = self.compute_torque(ctrl, i)
            elif cfg.action_type == 'torque':
                torque = ctrl[:self.ndof] * cfg.a_scale
            torque = np.clip(torque, -cfg.torque_lim, cfg.torque_lim)
            self.data.ctrl[:] = torque

            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[self.ndof: self.ndof+self.vf_dim].copy()
                self.rfc_implicit(vf)

            self.sim.step()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a):
        cfg = self.cfg
        # record prev state
        self.prev_qpos = self.data.qpos.copy()
        self.prev_qvel = self.data.qvel.copy()
        self.prev_bquat = self.bquat.copy()  # 计算body角速度用的.
        self.prev_skt_pos = self.skt_pos.copy()  # 计算body joint local速度用的.
        # self.qpos_ref = self.prev_qpos  # qpos_target太飘了... 但是用simulation current qpos会直接缩一团. 换previous qpos?
        self.qpos_ref = self.get_pose_target()  # qpos_target太飘了... 但是用current qpos会直接缩一团.
        # do simulation
        self.do_simulation(a, self.frame_skip)
        self.cur_t += 1
        self.bquat = self.get_body_quat()
        self.skt_pos = self.get_skeleton_pos(self.cfg.obs_coord)
        self.sync_expert()
        # get obs
        head_pos = self.get_body_com('Head')
        reward = 1.0
        if self.fix_head_lb is not None:
            fail = head_pos[2] < self.fix_head_lb
        else:
            # fail = self.expert is not None and head_pos[2] < self.expert['head_height_lb'] - 0.1
            # fail = self.expert is not None and head_pos[2] < self.get_expert_attr('head_pos', ind)[2] - 0.2
            ind = self.get_expert_index(self.cur_t)
            e_skt_wpos = self.get_expert_attr('skt_wpos', ind)  # 全身的joint world 坐标系.
            e_head_height = self.get_expert_attr('head_pos', ind)[2] - np.min(e_skt_wpos.reshape(16, 3)[:, 2])
            fail = self.expert is not None and head_pos[2] < e_head_height - 0.2

        end = self.cur_t >= (cfg.env_episode_len if self.fix_len is None else self.fix_len)
        done = fail or end
        return self.get_obs(), reward, done, {'fail': fail, 'end': end}

    def reset_model(self):  #
        if self.fix_start_state is not None:
            # init_pose = self.fix_start_state[:self.model.nq]
            # init_vel = self.fix_start_state[self.model.nq:]
            # self.set_state(init_pose, init_vel)
            cfg = self.cfg
            fr_margin = cfg.fr_margin
            # sample expert
            expert_ind = self.np_random.randint(len(self.expert_list)) if self.fix_expert_ind is None else self.fix_expert_ind
            self.set_expert(expert_ind)
            # sample fix frame 这个在我的实验设定还比较重要, 因为会出现炸了的qpos, 从videopose那边过来的.
            if self.fix_start_ind is None:
                ind = 0 if cfg.env_start_first else self.np_random.randint(fr_margin, self.expert['len'] - cfg.env_episode_len - fr_margin)
            else:
                ind = self.fix_start_ind
            self.start_ind = ind
            if hasattr(cfg, 'random_cur_t') and cfg.random_cur_t:
                self.cur_t = np.random.randint(cfg.env_episode_len)
                ind += self.cur_t
            init_pose = self.fix_start_state[:self.model.nq].copy()
            init_vel = self.fix_start_state[self.model.nq:].copy()
            init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
            self.set_state(init_pose, init_vel)  # 训练的时候直接设定当前expert的状态加噪音
            self.bquat = self.get_body_quat()
            self.sync_expert()
        elif self.expert_list is not None:
            assert False, 'this part may cause NAN due to the poor quality of predicted qpos'
            # cfg = self.cfg
            # fr_margin = cfg.fr_margin
            # # sample expert
            # expert_ind = self.np_random.randint(len(self.expert_list)) if self.fix_expert_ind is None else self.fix_expert_ind
            # self.set_expert(expert_ind)
            # # sample start frame 这个在我的实验设定还比较重要
            # if self.fix_start_ind is None:
            #     ind = 0 if cfg.env_start_first else self.np_random.randint(fr_margin, self.expert['len'] - cfg.env_episode_len - fr_margin)
            # else:
            #     ind = self.fix_start_ind
            # self.start_ind = ind
            # if hasattr(cfg, 'random_cur_t') and cfg.random_cur_t:
            #     self.cur_t = np.random.randint(cfg.env_episode_len)
            #     ind += self.cur_t
            # init_pose = self.expert['qpos'][ind, :].copy()
            # init_vel = self.expert['qvel'][ind, :].copy()
            # init_pose[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.model.nq - 7)
            # self.set_state(init_pose, init_vel)  # 训练的时候直接设定当前expert的状态加噪音
            # self.bquat = self.get_body_quat()
            # self.sync_expert()
        else:
            init_pose = self.data.qpos
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.data.qpos[0:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def set_fix_sampling(self, expert_ind=None, start_ind=None, len=None, start_state=None, cnn_feat=None):
        self.fix_expert_ind = expert_ind
        self.fix_start_ind = start_ind
        self.fix_len = len
        self.fix_start_state = start_state if self.fix_start_state is None else self.fix_start_state
        self.fix_cnn_feat = cnn_feat

    def set_fix_head_lb(self, fix_head_lb=None):
        self.fix_head_lb = fix_head_lb

    def sync_expert(self):
        if self.expert is not None and self.cur_t % self.cfg.sync_exp_interval == 0:
            expert = self.expert
            ind = self.get_expert_index(self.cur_t)
            e_qpos = self.get_expert_attr('qpos', ind).copy()
            expert['rel_heading'] = quaternion_multiply(get_heading_q(self.data.qpos[3:7]),
                                                        quaternion_inverse(get_heading_q(e_qpos[3:7])))
            expert['start_pos'] = e_qpos[:3]
            expert['sim_pos'] = np.concatenate((self.data.qpos[:2], np.array([e_qpos[2]])))

    def set_expert(self, expert_ind):
        self.expert_ind = expert_ind
        self.expert_id = self.expert_list[expert_ind]  # take id
        self.expert = self.expert_arr[expert_ind]  # take array dict

    def get_expert_index(self, t):
        return self.start_ind + t

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind, :]

    def get_pose_dist(self):
        ind = self.get_expert_index(self.cur_t)
        qpos_e = self.expert['qpos'][ind, :]
        qpos_g = self.data.qpos
        diff = qpos_e - qpos_g
        return np.linalg.norm(diff[2:])

    def get_pose_diff(self):
        ind = self.get_expert_index(self.cur_t)
        qpos_e = self.expert['qpos'][ind, :]
        qpos_g = self.data.qpos
        diff = qpos_e - qpos_g
        return np.abs(diff[2:])

    def get_pose_target(self):
        ind = self.get_expert_index(self.cur_t)
        qpos_target = self.expert['qpos'][ind+1, :] * 1.
        if self.cfg.arm_reset:
            # remove noise hand part
            # qpos_target[25:45] = self.cfg.a_ref[18:38] * 1.  # 这个会导致手部动作幅度不够大. 但是不要的话shoulder会转风车.
            qpos_target[28:31] = self.cfg.a_ref[21:24] * 1.  # 因为arm x计算的数值附加跳动180 error的, 所以这里归0
            qpos_target[38:41] = self.cfg.a_ref[31:34] * 1.  # 因为arm x计算的数值附加跳动180 error的, 所以这里归0
        if self.cfg.ref_reset:
            qpos_target[7:] = self.cfg.a_ref[:] * 1
        return qpos_target

    def get_poseseq_target(self, r=9):
        ind = self.get_expert_index(self.cur_t)
        qpos_target = self.expert['qpos'][ind-r+1:ind+1, :] * 1.
        qpos_target = qpos_target.reshape(-1)
        return qpos_target

    def get_episode_cnn_feat(self):  # 更新但前的cnn feature缓存， 送到RNN里面。
        fm = self.cfg.fr_margin
        num_fr = self.cfg.env_episode_len if self.fix_len is None else self.fix_len
        return self.cnn_feat[self.expert_ind][self.start_ind - fm: self.start_ind + num_fr + fm, :] \
            if self.fix_cnn_feat is None else self.fix_cnn_feat

    def get_episode_export_qpos_qvel(self):  # 用在eval的reset_env_state。但是由于可能NAN, 暂时放置.
        fm = self.cfg.fr_margin
        num_fr = self.cfg.env_episode_len if self.fix_len is None else self.fix_len
        # load qpos
        qpos = self.expert_arr[self.expert_ind]['qpos'][self.start_ind: self.start_ind + num_fr, 2:]  # no x y infor
        qvel = self.expert_arr[self.expert_ind]['qvel'][self.start_ind: self.start_ind + num_fr, :]
        return np.concatenate([qpos, qvel], axis=-1).copy()

        # return self.cnn_feat[self.expert_ind][self.start_ind - fm: self.start_ind + num_fr + fm, :] \
        #     if self.fix_cnn_feat is None else self.fix_cnn_feat



