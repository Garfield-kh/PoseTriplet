from utils import *



def quat_space_reward_v6(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv = ws.get('w_p', 0.5), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_rp', 0.1), ws.get(
        'w_rv', 0.1)
    k_p, k_v, k_e = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20)
    k_rh, k_rq, k_rl, k_ra = ws.get('k_rh', 300), ws.get('k_rq', 300), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)
    v_ord = ws.get('v_ord', 2)
    w_vf, k_vf = ws.get('w_vf', 0.0), ws.get('k_vf', 1)  # residual force
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat  
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)  
    cur_rlinv_local = cur_qvel[:3]  
    cur_rangv = cur_qvel[3:6]  
    cur_rq_rmh = de_heading(cur_qpos[3:7])  
    cur_ee = env.get_ee_pos(cfg.obs_coord)  
    cur_bquat = env.get_body_quat()  
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)  
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)  
    e_rangv = env.get_expert_attr('rangv', ind)  
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)  
    e_ee = env.get_expert_attr('ee_pos', ind)   
    e_bquat = env.get_expert_attr('bquat', ind)  
    e_bangvel = env.get_expert_attr('bangvel', ind)  
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))  # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # root position reward
    root_height_dist = cur_qpos[2] - e_qpos[2]  
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]  
    root_pose_reward = math.exp(-k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2))
    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)  
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)  
    root_vel_reward = math.exp(-k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2))
    # residual force reward
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0

    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + w_e * ee_reward + w_rp * root_pose_reward + w_rv * root_vel_reward + w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_rp + w_rv + w_vf
    if ws.get('decay', False):
        reward *= 1.0 - t / cfg.env_episode_len
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, vel_reward, ee_reward, root_pose_reward, root_vel_reward, vf_reward])



def quat_space_reward_v7(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights
    # w_p, w_v, w_e, w_rp, w_rv = ws.get('w_p', 0.5), ws.get('w_v', 0.1), ws.get('w_e', 0.2), ws.get('w_rp', 0.1), ws.get(
    #     'w_rv', 0.1)
    # k_p, k_v, k_e = ws.get('k_p', 2), ws.get('k_v', 0.005), ws.get('k_e', 20)
    # k_rh, k_rq, k_rl, k_ra = ws.get('k_rh', 300), ws.get('k_rq', 300), ws.get('k_rl', 5.0), ws.get('k_ra', 0.5)
    v_ord = ws.get('v_ord', 2)

    w_p, k_p = ws.get('w_p', 0.5), ws.get('k_p', 2)  # qpos
    w_v, k_v = ws.get('w_v', 0.1), ws.get('k_v', 0.005)  # qvel
    w_e, k_e = ws.get('w_e', 0.2), ws.get('k_e', 20)  # ee
    w_rh, k_rh, v_rh = ws.get('w_rh', 0.0), ws.get('k_rh', 300) , ws.get('v_rh', 0.1)  # root height

    w_rq, k_rq = ws.get('w_rq', 0.0), ws.get('k_rq', 1)  
    w_rp, k_rp = ws.get('w_rp', 0.0), ws.get('k_rp', 1)  # root position
    w_rlv, k_rlv = ws.get('w_rlv', 0.0), ws.get('k_rlv', 1)  # root linear v
    w_rav, k_rav = ws.get('w_rav', 0.0), ws.get('k_rav', 1)  # root angle v
    w_vf, k_vf = ws.get('w_vf', 0.0), ws.get('k_vf', 1)  # residual force
    w_bae, k_bae = ws.get('w_bae', 0.0), ws.get('k_bae', 1)  # bone angle velocity energy
    w_headh, k_headh = ws.get('w_headh', 0.0), ws.get('k_headh', 1)  # headhight

    
    w_flv, k_flv = ws.get('w_flv', 0.0), ws.get('k_flv', 1)  # feet linear vel
    w_frp, k_frp = ws.get('w_frp', 0.0), ws.get('k_frp', 1)  # feet linear vel

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat  
    prev_skt_pos = env.prev_skt_pos  
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)  
    cur_rlinv_local = cur_qvel[:3]  
    cur_rangv = cur_qvel[3:6]  
    cur_rq_rmh = de_heading(cur_qpos[3:7])  
    cur_ee = env.get_ee_pos(cfg.obs_coord)  
    cur_bquat = env.get_body_quat()  
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)  
    cur_skt_pos = env.get_skeleton_pos(cfg.obs_coord)
    cur_sktvel = get_sktvel_fd(prev_skt_pos, cur_skt_pos, env.dt)
    # expert
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)  
    e_rangv = env.get_expert_attr('rangv', ind)  
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)  
    e_ee = env.get_expert_attr('ee_pos', ind)   
    e_bquat = env.get_expert_attr('bquat', ind)  
    e_bangvel = env.get_expert_attr('bangvel', ind)  
    e_sktvel = env.get_expert_attr('sktvel', ind)  
    e_skt_wpos = env.get_expert_attr('skt_wpos', ind)  
    e_skt_pos = env.get_expert_attr('skt_pos', ind)  
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat[4:], e_bquat[4:]))  # ignore root
    pose_diff = pose_diff * cfg.b_diffw / np.sum(cfg.b_diffw)  
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_diff = cur_bangvel[3:] - e_bangvel[3:]
    vel_diff = vel_diff.reshape(-1, 3)
    vel_diff = vel_diff * np.expand_dims(cfg.b_diffw, -1) / np.sum(cfg.b_diffw * 3)
    vel_dist = np.linalg.norm(vel_diff.reshape(-1), ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # feet joint vel reward
    flv_dist = np.linalg.norm(cur_sktvel.reshape(16, 3)[[3, 6, 12, 15], :] - e_sktvel.reshape(16, 3)[[3, 6, 12, 15], :], ord=v_ord)  # ignore root
    flv_dist = min(1, flv_dist)  
    flv_reward = math.exp(-k_flv * (flv_dist ** 2))
    # feet relative position
    e_feet_vec = e_skt_pos.reshape(16, 3)[3] - e_skt_pos.reshape(16, 3)[6]
    c_feet_vec = cur_skt_pos.reshape(16, 3)[3] - cur_skt_pos.reshape(16, 3)[6]
    frp_dist = np.linalg.norm(c_feet_vec - e_feet_vec)
    frp_reward = math.exp(-k_frp * (frp_dist ** 2))
    # root height reward
    
    e_root_height = e_skt_wpos.reshape(16, 3)[0, 2] - np.min(e_skt_wpos.reshape(16, 3)[:, 2]) + v_rh #  + 0.08
    root_height_dist = cur_qpos[2] - e_root_height # e_qpos[2]  
    root_height_reward = math.exp(-k_rh * (root_height_dist ** 2))
    # root quat reward
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]  
    root_quat_reward = math.exp(- k_rq * (root_quat_dist ** 2))
    # root linear velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)  
    root_linv_dist = min(1, root_linv_dist)  
    root_linv_reward = math.exp(-k_rlv * (root_linv_dist ** 2))
    # root angler velocity reward
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)  
    root_angv_reward = math.exp(- k_rav * (root_angv_dist ** 2))
    # residual force reward
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_norm = np.linalg.norm(vf)
        vf_reward = math.exp(-k_vf * (vf_norm ** 2))
    else:
        vf_norm = 0.0
        vf_reward = 0.0

    # overall reward
    reward = w_p * pose_reward + w_v * vel_reward + \
             w_e * ee_reward + w_flv * flv_reward + \
             w_frp * frp_reward + \
             w_rh * root_height_reward + w_rq * root_quat_reward + \
             w_rlv * root_linv_reward + w_rav * root_angv_reward + \
             w_vf * vf_reward
    reward /= w_p + w_v + w_e + w_flv + w_frp + w_rh + w_rq + w_rlv + w_rav + w_vf
    if ws.get('decay', False):
        reward *= 1.0 - t / cfg.env_episode_len
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_reward, pose_dist,
                             vel_reward, vel_dist,
                             ee_reward, ee_dist,
                             flv_reward, flv_dist,
                             frp_reward, frp_dist,
                             root_height_reward, root_height_dist,
                             root_quat_reward, root_quat_dist,
                             root_linv_reward, root_linv_dist,
                             root_angv_reward, root_angv_dist,
                             vf_reward, vf_norm,
                             ])




########################################################################################
"""loop starting reward"""
########################################################################################


def followtraj_reward_v7(env, state, action, info):
    # reward coefficients
    cfg = env.cfg
    ws = cfg.reward_weights

    w_rq, k_rq = ws.get('w_rq', 0.0), ws.get('k_rq', 1)  
    w_rp, k_rp = ws.get('w_rp', 0.0), ws.get('k_rp', 1)  
    w_rlvw, k_rlvw1, k_rlvw2 = ws.get('w_rlvw', 0.0), ws.get('k_rlvw1', 1), ws.get('k_rlvw2', 1)  
    w_rlv, k_rlv = ws.get('w_rlv', 0.0), ws.get('k_rlv', 1)  # root linear v
    w_rav, k_rav = ws.get('w_rav', 0.0), ws.get('k_rav', 1)  # root angle v
    w_vf, k_vf = ws.get('w_vf', 0.0), ws.get('k_vf', 1)  # residual force
    w_dof, k_dof = ws.get('w_dof', 0.0), ws.get('k_dof', 1)  # residual force
    w_bae, k_bae = ws.get('w_bae', 0.0), ws.get('k_bae', 1)  # bone angle velocity energy
    w_headh, k_headh = ws.get('w_headh', 0.0), ws.get('k_headh', 1)  # headhight
    w_eeh, k_eeh = ws.get('w_eeh', 0.0), ws.get('k_eeh', 1)  # headhight

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat  
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_rlinv_world = get_qvel_fd(prev_qpos, cur_qpos, env.dt)[:3]  
    cur_qvel = get_qvel_fd(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)  
    cur_rlinv_local = cur_qvel[:3]  
    cur_rangv = cur_qvel[3:6]  
    cur_rq_rmh = de_heading(cur_qpos[3:7])  
    # cur_ee = env.get_ee_pos(cfg.obs_coord)  
    cur_sktwpose = env.get_skeleton_pos()  
    cur_bquat = env.get_body_quat()  
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)  
    # cur_eeh = env.get_ee_pos(cfg.obs_coord)[-3:]  
    cur_eeh = env.get_ee_pos(None)[-3:]  
    # expert
    target_qpos = env.get_expert_attr('qpos', -1)  
    e_qpos = env.get_expert_attr('qpos', ind)
    e_rlinv_local = env.get_expert_attr('rlinv_local', ind)  
    e_rangv = env.get_expert_attr('rangv', ind)  
    e_rq_rmh = env.get_expert_attr('rq_rmh', ind)  
    e_eeh = env.get_expert_attr('ee_wpos', ind)[-3:]  

    # head height rewards  
    head_height = env.get_body_com('Head')[2]
    head_height_reward = math.exp( - k_headh * (1/head_height))

    # posture energy
    bangvel_energy = np.mean(np.square(cur_bangvel))
    bangvel_energy_reward = math.exp( - k_bae * (1/bangvel_energy))

    # root position reward
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]  
    root_quat_reward = math.exp( - k_rq * (root_quat_dist ** 2))

    # root postion
    # target_position = e_qpos[:2] - cur_qpos[:2]  
    target_position = target_qpos[2] - cur_qpos[2]
    root_position_dist = np.linalg.norm(target_position)
    root_position_reward = math.exp(-k_rp * (root_position_dist ** 2))
    # root_position_reward = math.exp(-k_rp * root_position_dist)
    root_linvw_dist = np.linalg.norm(k_rlvw1 * cur_rlinv_world[:2])
    root_linvw_reward = math.exp(-k_rlvw2 * (root_linvw_dist ** 2))

    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv[2:] - e_rangv[2:])
    root_linv_reward = math.exp(-k_rlv * (root_linv_dist ** 2))
    root_angv_reward = math.exp(-k_rav * (root_angv_dist ** 2))

    # end point head reward
    root_eeh_dist = np.linalg.norm(cur_eeh[2] - e_eeh[2])
    root_eeh_reward = math.exp(-k_eeh * (root_eeh_dist ** 2))

    # joint force reward
    # dof_ = action[:env.ndof]
    # dof = env.data.ctrl
    # dof_reward = math.exp(-k_dof * (np.linalg.norm(dof) ** 2))

    # residual force reward
    if w_vf > 0.0:
        vf = action[-env.vf_dim:]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf = 0
        vf_reward = 0.0

    # overall reward
    reward = w_rq * root_quat_reward + w_rp * root_position_reward + w_rlvw * root_linvw_reward \
             + w_rlv * root_linv_reward + w_rav * root_angv_reward + \
             w_vf * vf_reward + w_bae * bangvel_energy_reward \
             + w_headh * head_height_reward + w_eeh * root_eeh_reward
    reward /= w_rq + w_rp + w_rlvw + w_rlv + w_rav + w_vf + w_bae + w_headh + w_eeh
    if ws.get('decay', False):
        reward *= 1.0 - t / cfg.env_episode_len
    if info['end']:
        reward += env.end_reward
    return reward, np.array([root_quat_reward, root_quat_dist,
                             root_position_reward, root_position_dist,
                             # root_linvw_reward, root_linvw_dist,
                             root_linv_reward, root_linv_dist,
                             root_angv_reward, root_angv_dist,
                             # vf_reward, np.linalg.norm(vf),
                             bangvel_energy_reward, bangvel_energy,
                             head_height_reward, head_height,
                             root_eeh_reward, root_eeh_dist,
                             ])




def constant_reward(env, state, action, info):
    reward = 1.0
    if info['end']:
        reward += env.end_reward
    return 1.0, np.zeros(1)


def pose_dist_reward(env, state, action, info):
    pose_dist = env.get_pose_dist()
    reward = 5.0 - 3.0 * pose_dist
    if info['end']:
        reward += env.end_reward
    return reward, np.array([pose_dist])

#################################################################
#################################################################


reward_func = {
    'quat_v6': quat_space_reward_v6,
    'quat_v7': quat_space_reward_v7,
    'followtraj_reward_v7': followtraj_reward_v7,
    'constant': constant_reward,
    'pose_dist': pose_dist_reward,
}

reward_name_list = {
    'quat_v6': ['pose_reward', 'vel_reward', 'ee_reward', 'root_pose_reward', 'root_vel_reward', 'vf_reward'],
    'quat_v7': ['pose_reward', 'pose_dist',
                 'vel_reward', 'vel_dist',
                 'ee_reward', 'ee_dist',
                'flv_reward', 'flv_dist',
                'frp_reward', 'frp_dist',
                'root_height_reward', 'root_height_dist',
                 'root_quat_reward', 'root_quat_dist',
                 'root_linv_reward', 'root_linv_dist',
                 'root_angv_reward', 'root_angv_dist',
                 'vf_reward', 'vf_norm',
                             ],
    'followtraj_reward_v7': ['root_quat_reward', 'root_quat_dist',
                             'root_height_reward', 'root_height_dist',
                             # 'root_linvw_reward', 'root_linvw_dist',
                             'root_linv_reward', 'root_linv_dist',
                             'root_angv_reward', 'root_angv_dist',
                             # 'vf_reward', 'vf_norm',
                             'bangvel_energy_reward', 'bangvel_energy',
                             'head_height_reward', 'head_height',
                             'root_eeh_reward', 'root_eeh_dist',
                             ],
    'constant': ['constant_reward'],
    'pose_dist': ['pose_dist_reward'],
}
