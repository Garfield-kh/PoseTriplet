# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as path
import pickle
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from common.camera import *
from common.generators import ChunkedGenerator, UnchunkedGenerator, ChunkedNoPadGeneratorV5
from common.loss import *
from common.model import *
from common.utils import deterministic_random
from function.gan_utils import pose_seq_bl_aug
from function.logger import create_logger
from function.utils import Summary, get_scheduler, get_discriminator_accuracy, Sample_from_Pool, get_contacts

'''
basement class
'''


class PoseGANBasement(object):
    def __init__(self, args):
        # init param
        self.device = torch.device("cuda")
        self.args = args

        # define checkpoint directory # Create checkpoint directory if it does not exist
        # self.args.checkpoint = path.join(self.args.checkpoint, self.args.note,
        #                                  datetime.datetime.now().strftime('%m%d%H%M%S'))
        self.args.checkpoint = path.join(self.args.checkpoint, self.args.note)
        print('INFO: creat log folder at {}'.format(self.args.checkpoint))
        os.makedirs(self.args.checkpoint, exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoint, 'ckpt'), exist_ok=True)

        # prepare monitor
        # Init monitor for net work training
        #########################################################
        self.summary = Summary(self.args.checkpoint)
        self.writer = self.summary.create_summary()
        self.logger = create_logger(os.path.join(self.args.checkpoint, 'log.txt'))
        self.logger.info(args)

    def logging(self, val_rlt, epoch_start_time):
        """
        recording the process of posenet, and saving model.
        """
        lr = self.optimizer_P.param_groups[0]['lr']
        losses_str = ' '.join(['{}: {:.4f}'.format(val_set_key, val_rlt[val_set_key]['p1']) \
                               for val_set_key in val_rlt])
        dt = (time() - epoch_start_time) / 60
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} min {} lr: {:.5f}'.format(self.summary.epoch, dt, losses_str, lr))

        # record the result list and save ckpt
        if self.summary.epoch <= 2:
            self.h36m_p1_s911 = []
        self.h36m_p1_s911.append(val_rlt['s911']['p1'])
        # Save checkpoint if necessary
        if self.h36m_p1_s911[-1] == min(self.h36m_p1_s911):
            ckpt_path = os.path.join(self.args.checkpoint, 'ckpt', 'best_ckpt_S911.bin')
            self.logger.info('Saving checkpoint to{}'.format(ckpt_path))
            torch.save({
                'model_pos': self.model_pos_train.state_dict(),
                'model_traj': self.model_traj_train.state_dict(),
            }, ckpt_path)

        if self.summary.epoch % 5 == 0:
            ckpt_path = os.path.join(self.args.checkpoint, 'ckpt', 'ckpt_ep_{:0>3d}.bin'.format(self.summary.epoch))
            self.logger.info('Saving checkpoint to{}'.format(ckpt_path))
            torch.save({
                'model_pos': self.model_pos_train.state_dict(),
                'model_traj': self.model_traj_train.state_dict(),
            }, ckpt_path)

    def data_preparation(self):
        ###################################
        # prepare data
        ###################################
        self.logger.info('Loading dataset...')
        dataset_path = 'data/data_3d_' + self.args.dataset + '.npz'
        if self.args.dataset == 'h36m':
            from common.h36m_dataset import Human36mDataset
            self.dataset = Human36mDataset(dataset_path)
        else:
            raise KeyError('Invalid dataset')

        self.logger.info('Preparing dataset...')
        for subject in self.dataset.subjects():
            for action in self.dataset[subject].keys():
                anim = self.dataset[subject][action]

                if 'positions' in anim:
                    positions_3d = []
                    for cam in anim['cameras']:
                        pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        # pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position, no, keep here
                        positions_3d.append(pos_3d)  # T x J x 3
                    anim['positions_3d'] = positions_3d

                if 'positions' in anim:
                    contact_labels = []
                    for cam in anim['cameras']:
                        contact_label = get_contacts(anim['positions'])
                        contact_labels.append(contact_label)  # T x 2 x 1
                    anim['contact_labels'] = contact_labels

        self.keypoints_preparation()

    def keypoints_preparation(self):
        # 2D keypoint
        self.logger.info('Loading 2D detections...')
        self.keypoints = np.load('data/data_2d_' + self.args.dataset + '_' + self.args.keypoints + '.npz',
                                 allow_pickle=True)
        # keypoints_metadata = self.keypoints['metadata'].item()
        keypoints_metadata = {'num_joints': 16,
                              'keypoints_symmetry': [[4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]]}

        keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(self.dataset.skeleton().joints_left()), list(
            self.dataset.skeleton().joints_right())
        self.keypoints = self.keypoints['positions_2d'].item()

        for subject in self.dataset.subjects():
            assert subject in self.keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in self.dataset[subject].keys():
                assert action in self.keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                if 'positions_3d' not in self.dataset[subject][action]:
                    continue

                for cam_idx in range(len(self.keypoints[subject][action])):

                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = self.dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert self.keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if self.keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        self.keypoints[subject][action][cam_idx] = self.keypoints[subject][action][cam_idx][
                                                                   :mocap_length]

                assert len(self.keypoints[subject][action]) == len(self.dataset[subject][action]['positions_3d'])

        # norm keypoint
        for subject in self.keypoints.keys():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    # Normalize camera frame
                    cam = self.dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    self.keypoints[subject][action][cam_idx] = kps

    def fetch(self, subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        out_camera_rtparams = []
        out_contact_labels = []
        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                if subject in self.dataset.cameras():
                    cams = self.dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])
                        if 'extrinsic' in cam:
                            out_camera_rtparams.append(cam['extrinsic'])

                if parse_3d_poses and 'positions_3d' in self.dataset[subject][action]:
                    poses_3d = self.dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):  # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])

                # for contact labels, same as poses_3d
                if parse_3d_poses and 'contact_labels' in self.dataset[subject][action]:
                    contact_labels = self.dataset[subject][action]['contact_labels']
                    assert len(contact_labels) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(contact_labels)):  # Iterate across cameras
                        out_contact_labels.append(contact_labels[i])

        if len(out_camera_params) == 0:
            assert False
        if len(out_camera_rtparams) == 0:
            assert False
        if len(out_poses_3d) == 0:
            assert False
        if len(out_contact_labels) == 0:
            assert False

        stride = self.args.downsample
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
                    out_contact_labels[i] = out_contact_labels[i][::stride]

        return out_camera_params, out_camera_rtparams, out_poses_3d, out_poses_2d, out_contact_labels

    def dataloader_preparation(self):
        action_filter = None if self.args.actions == '*' else self.args.actions.split(',')
        if action_filter is not None:
            self.logger.info('Selected actions:{}'.format(action_filter))

        ###################################
        # train subject # test subject
        ###################################
        subjects_train = self.args.subjects_train.split(',')
        subjects_test = self.args.subjects_test.split(',')

        cameras_train, cam_rt_train, poses_train, poses_train_2d, contact_train = self.fetch(subjects_train,
                                                                                             action_filter,
                                                                                             subset=self.args.subset)
        cameras_valid, cam_rt_valid, poses_valid, self.poses_valid_2d, contact_valid = self.fetch(subjects_test,
                                                                                                  action_filter)
        causal_shift = 0
        self.pad = (np.prod([int(x) for x in self.args.architecture.split(',')]) - 1) // 2
        self.rf = np.prod([int(x) for x in self.args.architecture.split(',')])


        ##################################################################
        ##### linkstart: load expert,
        ##################################################################
        self.logger.info('INFO: self.args.expert_dict_path: {}'.format(self.args.expert_dict_path))
        if self.args.expert_dict_path is None:
            expert_dict = {'h36m_take_000':{'skt_wpos':np.ones((500, 16, 3))}}
            take_list = ['h36m_take_000']
        else:
            expert_feat_file = self.args.expert_dict_path
            expert_dict = pickle.load(open(expert_feat_file, 'rb'))
            # take_list = [take for take in expert_dict if expert_dict[take]['t_num_reset'] == 0] # maybe filter some
            take_list = ['h36m_take_{:0>3d}'.format(i) for i in range(600)]

        # load expert from rib-rl
        self.logger.info('INFO: self.args.extra_expert_dict_path: {}'.format(self.args.extra_expert_dict_path))
        if self.args.extra_expert_dict_path is None:
            extra_expert_dict = None
            extra_take_list = []
        else:
            extra_expert_feat_file = self.args.extra_expert_dict_path
            extra_expert_dict = pickle.load(open(extra_expert_feat_file, 'rb'))
            extra_take_list = [take for take in extra_expert_dict if extra_expert_dict[take]['t_num_reset'] == 0]

        ######################################################################
        # prepare a basement for every epoch update
        self.fixed_fake_database = {
            'cam_rt_train': cam_rt_train,
            'cameras_train': cameras_train,
            'take_list': take_list,
            'expert_dict': expert_dict,
            'causal_shift': causal_shift,
            'extra_expert_dict': extra_expert_dict,
            'extra_take_list': extra_take_list,
        }


        skt_pos_train = []
        skt_pos_train_2dtarget = []

        for i, take in enumerate(take_list):
            # assume expert is less and shorter than h36m.
            skt_pos_train.append(world2cam_sktpos(expert_dict[take]['skt_wpos']))
            # skt_pos_train.append(world2cam_sktpos(reset_spine(expert_dict[take]['skt_wpos'])))
            skt_pos_train_2dtarget.append(poses_train_2d[i][10:expert_dict[take]['skt_wpos'].shape[0] + 10])

        ########################################################################################
        # extra
        for i, take in enumerate(extra_take_list):
            # assume expert is less and shorter than h36m.
            skt_pos_train.append(world2cam_sktpos(extra_expert_dict[take]['skt_wpos']))
            # skt_pos_train.append(world2cam_sktpos(reset_spine(extra_expert_dict[take]['skt_wpos'])))
            skt_pos_train_2dtarget.append(poses_train_2d[i%len(poses_train_2d)][10:extra_expert_dict[take]['skt_wpos'].shape[0] + 10])
        ########################################################################################

        # prepare data for augmenting
        aug_pad = self.pad
        self.aug_generator = ChunkedNoPadGeneratorV5(self.args.batch_size // self.args.stride, None, None,
                                                     skt_pos_train, skt_pos_train_2dtarget, None, self.args.stride,
                                                     pad=aug_pad, causal_shift=causal_shift, shuffle=True,
                                                     # augment=True,
                                                     augment=self.args.data_augmentation,
                                                     kps_left=self.kps_left, kps_right=self.kps_right,
                                                     joints_left=self.joints_left, joints_right=self.joints_right)
        self.logger.info('INFO: aug-supervision on {} frames'.format(self.aug_generator.num_frames()))
        self.fake_cam_sample = Sample_from_Pool(max_elements=self.args.batch_size)

        # train loader s15678 eval
        train_generator_eval = UnchunkedGenerator(cam_rt_train, poses_train, poses_train_2d,
                                                  pad=self.pad, causal_shift=causal_shift, augment=False,
                                                 kps_left=self.kps_left, kps_right=self.kps_right,
                                                 joints_left=self.joints_left,
                                                 joints_right=self.joints_right)
        self.logger.info('INFO: Testing on {} frames > train_generator_eval'.format(train_generator_eval.num_frames()))
        train_generator_eval_flip = UnchunkedGenerator(cam_rt_train, poses_train, poses_train_2d,
                                                  pad=self.pad, causal_shift=causal_shift, augment=True,
                                                 kps_left=self.kps_left, kps_right=self.kps_right,
                                                 joints_left=self.joints_left,
                                                 joints_right=self.joints_right)
        self.logger.info('INFO: Testing on {} frames > train_generator_eval_flip'.format(train_generator_eval_flip.num_frames()))
        # test loader -- S911
        test_generator_s911 = UnchunkedGenerator(None, poses_valid, self.poses_valid_2d,
                                                 pad=self.pad, causal_shift=causal_shift, augment=False,
                                                 kps_left=self.kps_left, kps_right=self.kps_right,
                                                 joints_left=self.joints_left,
                                                 joints_right=self.joints_right)
        self.logger.info('INFO: Testing on {} frames > test_generator_s911'.format(test_generator_s911.num_frames()))
        test_generator_s911_flip = UnchunkedGenerator(None, poses_valid, self.poses_valid_2d,
                                                 pad=self.pad, causal_shift=causal_shift, augment=True,
                                                 kps_left=self.kps_left, kps_right=self.kps_right,
                                                 joints_left=self.joints_left,
                                                 joints_right=self.joints_right)
        self.logger.info('INFO: Testing on {} frames > test_generator_s911_flip'.format(test_generator_s911_flip.num_frames()))

        # test loader  -- 3DHP # all frame are used.
        pkl_path = './data_cross/3dhp/3dhp_testset_bySub.pkl'
        test_generator_3dhp = self._dataloader_preparation(pkl_path=pkl_path,
                                                           key_2d='valid_kps_2d_imgnorm',
                                                           key_3d='valid_kps_3d',
                                                           clip_flg=True)
        test_generator_3dhp_flip = self._dataloader_preparation(pkl_path=pkl_path,
                                                                key_2d='valid_kps_2d_imgnorm',
                                                                key_3d='valid_kps_3d',
                                                                clip_flg=True,
                                                                test_augment=True)
        # test loader  -- 3DPWD
        pkl_path = './data_cross/3dpw/3dpw_testset_bySub.pkl'
        test_generator_3dpw = self._dataloader_preparation(pkl_path=pkl_path,
                                                           key_2d='joints_2d_imgnorm',
                                                           key_3d='valid_kps_3d',
                                                           clip_flg=True)
        test_generator_3dpw_flip = self._dataloader_preparation(pkl_path=pkl_path,
                                                                key_2d='joints_2d_imgnorm',
                                                                key_3d='valid_kps_3d',
                                                                clip_flg=True,
                                                                test_augment=True)

        ############################
        ## place all test loader together
        ############################
        self.val_generator_dict = {
            's15678': train_generator_eval,
            's15678_flip': train_generator_eval_flip,
            's911': test_generator_s911,
            's911_flip': test_generator_s911_flip,
            '3dhp': test_generator_3dhp,
            '3dhp_flip': test_generator_3dhp_flip,
            '3dpw': test_generator_3dpw,
            '3dpw_flip': test_generator_3dpw_flip,
        }

    def _dataloader_preparation(self, pkl_path, key_2d, key_3d, clip_flg, scale2d=1., test_augment=False):
        """
        dataloader for cross data
        """
        with open(pkl_path, 'rb') as fp:
            self.logger.info('load from pickle file -> {}'.format(pkl_path))
            tmp_npdict = pickle.load(fp)
        poses_3d = []
        poses_2d = []
        # clip_flg = True
        # [..., :2] for 2D is to remove the confidence channel.
        for sub in tmp_npdict:
            if clip_flg:
                for clip_idx in tmp_npdict[sub]['clip_idx']:
                    poses_3d.append(tmp_npdict[sub][key_3d][clip_idx[0]:clip_idx[1]])
                    poses_2d.append(tmp_npdict[sub][key_2d][clip_idx[0]:clip_idx[1]][..., :2] * scale2d)
            else:
                poses_3d.append(tmp_npdict[sub][key_3d])
                poses_2d.append(tmp_npdict[sub][key_2d][..., :2] * scale2d)

        test_generator = UnchunkedGenerator(cameras=None, poses_3d=poses_3d, poses_2d=poses_2d,
                                            pad=self.pad, causal_shift=0, augment=test_augment,
                                            kps_left=self.kps_left, kps_right=self.kps_right,
                                            joints_left=self.joints_left,
                                            joints_right=self.joints_right)
        self.logger.info('INFO: Testing on {} frames'.format(test_generator.num_frames()))
        return test_generator

    def s911_detect2d_dataloader_preparation(self):
        for det2d in ['hr']:
            self.logger.info('INFO: load s911 det2d: {}'.format(det2d))
            self.args.keypoints = det2d
            self.keypoints_preparation()
            self._s911_detect2d_dataloader_preparation(det2d)

    def _s911_detect2d_dataloader_preparation(self, det2d):
        causal_shift = 0
        action_filter = None if self.args.actions == '*' else self.args.actions.split(',')
        subjects_test = self.args.subjects_test.split(',')

        cameras_valid, cam_rt_valid, poses_valid, poses_valid_2d, contact_valid = self.fetch(subjects_test,
                                                                                                  action_filter)

        # test loader -- S911
        test_generator_s911 = UnchunkedGenerator(None, poses_valid, poses_valid_2d,
                                                 pad=self.pad, causal_shift=causal_shift, augment=False,
                                                 kps_left=self.kps_left, kps_right=self.kps_right,
                                                 joints_left=self.joints_left,
                                                 joints_right=self.joints_right)
        self.logger.info('INFO: Testing on {} frames > test_generator_s911 > det2d:{}'.format(test_generator_s911.num_frames(), det2d))
        test_generator_s911_flip = UnchunkedGenerator(None, poses_valid, poses_valid_2d,
                                                      pad=self.pad, causal_shift=causal_shift, augment=True,
                                                      kps_left=self.kps_left, kps_right=self.kps_right,
                                                      joints_left=self.joints_left,
                                                      joints_right=self.joints_right)
        self.logger.info(
            'INFO: Testing on {} frames > test_generator_s911_flip > det2d:{}'.format(test_generator_s911_flip.num_frames(), det2d))

        self.val_generator_dict['S911_{}'.format(det2d)] =  test_generator_s911
        self.val_generator_dict['S911_flip_{}'.format(det2d)] =  test_generator_s911_flip

    def update_fixedfake_train_generator(self):
        """
        update dataloader for each epoch
        include bone length augmentation and z-axis rotation
        """
        cam_rt_train = self.fixed_fake_database['cam_rt_train']
        cameras_train = self.fixed_fake_database['cameras_train']
        take_list = self.fixed_fake_database['take_list']
        expert_dict = self.fixed_fake_database['expert_dict']
        causal_shift = self.fixed_fake_database['causal_shift']
        extra_expert_dict = self.fixed_fake_database['extra_expert_dict']  # extra for boost exp
        extra_take_list = self.fixed_fake_database['extra_take_list']  # extra for boost exp

        fixed_fake_cam_rt_train = []
        fixed_fake_poses_train = []
        fixed_fake_poses_train_2d = []
        for i, take in enumerate(take_list):
            cam_ex = cam_rt_train[i]
            fixed_fake_cam_rt_train.append(cam_ex)

            tmp_skt_wpos = expert_dict[take]['skt_wpos'].reshape(-1, 16, 3).astype('float32')
            tmp_skt_wpos = zaxis_randrotation(tmp_skt_wpos)
            tmp_skt_wpos = pose_seq_bl_aug(torch.from_numpy(tmp_skt_wpos)).numpy()
            fixed_fake_poses_camed = world_to_camera_sktpos_v3(tmp_skt_wpos, self.args)
            fixed_fake_poses_train.append(fixed_fake_poses_camed)
            cam_ix = cameras_train[i]
            cam_ix_tf = torch.from_numpy(np.tile(cam_ix, (fixed_fake_poses_camed.shape[0], 1)))
            fixed_fake_poses_train_2d.append(
                project_to_2d_purelinear(fixed_fake_poses_camed))

        ############################################################
        # extra
        for i, take in enumerate(extra_take_list):
            i = i % len(cam_rt_train)
            cam_ex = cam_rt_train[i]
            fixed_fake_cam_rt_train.append(cam_ex)

            tmp_skt_wpos = extra_expert_dict[take]['skt_wpos'].reshape(-1, 16, 3).astype('float32')
            tmp_skt_wpos = zaxis_randrotation(tmp_skt_wpos)
            tmp_skt_wpos = pose_seq_bl_aug(torch.from_numpy(tmp_skt_wpos)).numpy()
            fixed_fake_poses_camed = world_to_camera_sktpos_v3(tmp_skt_wpos, self.args)
            fixed_fake_poses_train.append(fixed_fake_poses_camed)
            cam_ix = cameras_train[i]
            cam_ix_tf = torch.from_numpy(np.tile(cam_ix, (fixed_fake_poses_camed.shape[0], 1)))
            fixed_fake_poses_train_2d.append(
                project_to_2d_purelinear(fixed_fake_poses_camed))
        ########################################################################################

        self.train_generator = ChunkedGenerator(self.args.batch_size // self.args.stride, None,
                                                fixed_fake_poses_train,
                                                fixed_fake_poses_train_2d, self.args.stride,
                                                pad=self.pad, causal_shift=causal_shift, shuffle=True,
                                                augment=self.args.data_augmentation,
                                                # augment=False,
                                                kps_left=self.kps_left, kps_right=self.kps_right,
                                                joints_left=self.joints_left, joints_right=self.joints_right)
        self.logger.info('INFO: Training on {} frames'.format(self.train_generator.num_frames()))

    def _count_param(self, model, name):
        # print param number size.
        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        self.logger.info('INFO: Trainable parameter count for model {} is:{}'.format(name, model_params))

    def _model_preparation_pos(self):
        ######################################
        # prepare model: posenet: 2d pose -> 3d pose
        ######################################
        filter_widths = [int(x) for x in self.args.architecture.split(',')]
        if not self.args.disable_optimizations and not self.args.dense and self.args.stride == 1:
            # Use optimized model for single-frame predictions
            self.model_pos_train = TemporalModelOptimized1f(self.poses_valid_2d[0].shape[-2],
                                                            self.poses_valid_2d[0].shape[-1],
                                                            self.dataset.skeleton().num_joints(),
                                                            filter_widths=filter_widths, causal=self.args.causal,
                                                            dropout=self.args.dropout, channels=self.args.channels)
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            self.model_pos_train = TemporalModel(self.poses_valid_2d[0].shape[-2],
                                                 self.poses_valid_2d[0].shape[-1],
                                                 self.dataset.skeleton().num_joints(),
                                                 filter_widths=filter_widths, causal=self.args.causal,
                                                 dropout=self.args.dropout, channels=self.args.channels,
                                                 dense=self.args.dense)
        # model for eval
        self.model_pos = TemporalModel(self.poses_valid_2d[0].shape[-2],
                                       self.poses_valid_2d[0].shape[-1],
                                       self.dataset.skeleton().num_joints(),
                                       filter_widths=filter_widths, causal=self.args.causal, dropout=self.args.dropout,
                                       channels=self.args.channels, dense=self.args.dense)

        ##################################
        ##################################

        receptive_field = self.model_pos.receptive_field()
        self.logger.info('INFO: Receptive field: {} frames'.format(receptive_field))
        pad_check = (receptive_field - 1) // 2  # Padding on each side
        assert pad_check == self.pad, 'pad mismatch'

        # print param number size.
        self._count_param(self.model_pos_train, 'self.model_pos_train')

        self.model_pos = self.model_pos.cuda()
        self.model_pos_train = self.model_pos_train.cuda()

        ###################################
        # optimizer.
        ###################################
        self.optimizer_P = torch.optim.Adam(self.model_pos_train.parameters(), lr=self.args.learning_rate)

        self.lr_scheduler_P = get_scheduler(self.optimizer_P, policy='lambda', nepoch_fix=0, nepoch=self.args.epochs)

        ###################################
        # load pretrain
        ###################################
        if self.args.pretrain:
            ckpt_path = self.args.evaluate
            self.logger.info('Loading checkpoint at {}'.format(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model_pos_train.load_state_dict(checkpoint['model_pos'])
            self.model_pos.load_state_dict(checkpoint['model_pos'])

    def _model_preparation_traj(self):
        ######################################
        # prepare: posenet: 2d pose -> 3d traj
        ######################################
        filter_widths = [int(x) for x in self.args.architecture.split(',')]
        if not self.args.disable_optimizations and not self.args.dense and self.args.stride == 1:
            # Use optimized model for single-frame predictions
            self.model_traj_train = TemporalModelOptimized1f(self.poses_valid_2d[0].shape[-2],
                                                             self.poses_valid_2d[0].shape[-1],
                                                             1,
                                                             filter_widths=filter_widths, causal=self.args.causal,
                                                             dropout=self.args.dropout, channels=self.args.channels)
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            self.model_traj_train = TemporalModel(self.poses_valid_2d[0].shape[-2],
                                                  self.poses_valid_2d[0].shape[-1],
                                                  1,
                                                  filter_widths=filter_widths, causal=self.args.causal,
                                                  dropout=self.args.dropout, channels=self.args.channels,
                                                  dense=self.args.dense)
        # model for eval
        self.model_traj = TemporalModel(self.poses_valid_2d[0].shape[-2],
                                        self.poses_valid_2d[0].shape[-1],
                                        1,
                                        filter_widths=filter_widths, causal=self.args.causal, dropout=self.args.dropout,
                                        channels=self.args.channels, dense=self.args.dense)

        ##################################
        ##################################

        receptive_field = self.model_traj.receptive_field()
        self.logger.info('INFO: Receptive field: {} frames'.format(receptive_field))
        pad_check = (receptive_field - 1) // 2  # Padding on each side
        assert pad_check == self.pad, 'pad mismatch'

        # print param number size.
        self._count_param(self.model_traj_train, 'self.model_traj_train')

        self.model_traj = self.model_traj.cuda()
        self.model_traj_train = self.model_traj_train.cuda()

        ###################################
        # optimizer.
        ###################################
        self.optimizer_T = torch.optim.Adam(self.model_traj_train.parameters(), lr=self.args.learning_rate)

        self.lr_scheduler_T = get_scheduler(self.optimizer_T, policy='lambda', nepoch_fix=0, nepoch=self.args.epochs)

        ###################################
        # load pretrain
        ###################################
        if self.args.pretrain:
            ckpt_path = self.args.evaluate
            self.logger.info('Loading checkpoint at {}'.format(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model_traj_train.load_state_dict(checkpoint['model_traj'])
            self.model_traj.load_state_dict(checkpoint['model_traj'])

    def _model_preparation_Gcam(self):
        ######################################
        # prepare model: Gcam: 3d pose -> 3d pose, 2d pose, different cam.
        ######################################
        if self.args.gcam_choice == 'gcam_v0':
            from poseaugtool.model_virtualCam.virtualCam import G_camera
            self.model_Gcam = G_camera(self.args)
        # elif self.args.gcam_choice == 'gcam_v2':
        #     from poseaugtool.model_virtualCam.virtualCam import G_camera_v2
        #     self.model_Gcam = G_camera_v2(self.args)

        filter_ch = [int(x) for x in self.args.Dcamarchitecture.split(',')]
        # if self.args.dcam_choice == 'dcam_v0':
        #     from poseaugtool.model_virtualCam.virtualCam import Pose2DVideoDiscriminator
        #     self.model_Dcam = Pose2DVideoDiscriminator(ks=self.args.dcam_ks, nh_conv1d=filter_ch).to(self.device)
        # elif self.args.dcam_choice == 'dcam_v2':
        #     from poseaugtool.model_virtualCam.virtualCam import Pose2DVideoDiscriminatorV2
        #     self.model_Dcam = Pose2DVideoDiscriminatorV2(ks=self.args.dcam_ks, nh_conv1d=filter_ch).to(self.device)
        if self.args.dcam_choice == 'dcam_pa1':
            from poseaugtool.model_virtualCam.virtualCam import Pos2dPairDiscriminator
            self.model_Dcam = Pos2dPairDiscriminator().to(self.device)
        # elif self.args.dcam_choice == 'dcam_v5':
        #     from poseaugtool.model_virtualCam.virtualCam import Pos2dPairDiscriminator_v5
        #     self.model_Dcam = Pos2dPairDiscriminator_v5().to(self.device)
        # elif self.args.dcam_choice == 'dcam_v6':
        #     from poseaugtool.model_virtualCam.virtualCam import Pos2dPairDiscriminator_v6
        #     self.model_Dcam = Pos2dPairDiscriminator_v6().to(self.device)

        # print param number size.
        self._count_param(self.model_Gcam, 'self.model_Gcam')
        self._count_param(self.model_Dcam, 'self.model_Dcam')
        # to cuda
        self.model_Gcam = self.model_Gcam.cuda()
        self.model_Dcam = self.model_Dcam.cuda()
        ###################################
        # optimizer.
        ###################################
        self.optimizer_Gcam = optim.Adam(self.model_Gcam.parameters(),
                                         lr=self.args.lrgcam)  # , amsgrad=True)  #
        self.lr_scheduler_Gcam = get_scheduler(self.optimizer_Gcam, policy='lambda', nepoch_fix=0,
                                               nepoch=self.args.epochs)

        self.optimizer_Dcam = optim.Adam(self.model_Dcam.parameters(),
                                         lr=self.args.lrdcam)  # , amsgrad=True)  #
        self.lr_scheduler_Dcam = get_scheduler(self.optimizer_Dcam, policy='lambda', nepoch_fix=0,
                                               nepoch=self.args.epochs)
        ###################################
        # load pretrain
        ###################################
        if self.args.pretrain:
            pass

    def _train_batch_posenet(self, inputs_2d, inputs_3d, epoch_loss_3d_train, N):
        # here 3D shape is single frame. BxTxJx3: T=1
        target_3d_pose = inputs_3d[:, :, :, :] - inputs_3d[:, :, :1, :]
        # pos_3d[:, 1:] -= inputs_3d[:, :1]

        # Predict 3D poses
        predicted_3d_pos = self.model_pos_train(inputs_2d)

        self.optimizer_P.zero_grad()

        # loss_3d_pos = mpjpe(predicted_3d_pos, target_3d_pose)
        loss_3d_pos = self.MSE(predicted_3d_pos, target_3d_pose)
        loss_total = loss_3d_pos * 1.

        loss_total.backward()
        nn.utils.clip_grad_norm_(self.model_pos_train.parameters(),
                                 max_norm=1)
        self.optimizer_P.step()

        epoch_loss_3d_train += target_3d_pose.shape[0] * target_3d_pose.shape[1] * loss_3d_pos.item()
        N += target_3d_pose.shape[0] * target_3d_pose.shape[1]

        return loss_3d_pos.detach(), epoch_loss_3d_train, N

    def _train_batch_trajnet(self, inputs_2d, inputs_3d, epoch_loss_3d_train, N):
        target_3d_traj = inputs_3d[:, :, :1, :] * 1.   # focus on root traj.
        # pos_3d[:, 1:] -= inputs_3d[:, :1]

        # Predict 3D trajs
        predicted_3d_traj = self.model_traj_train(inputs_2d)
        # loss_3d_traj = mpjpe(predicted_3d_traj, target_3d_traj)

        self.optimizer_T.zero_grad()

        # loss_3d_traj = self.MSE(predicted_3d_traj, target_3d_traj)
        # weighted traj loss from videopose
        w = 1 / target_3d_traj[:, :, :, 2]  # Weight inversely proportional to depth
        loss_3d_traj = weighted_mpjpe(predicted_3d_traj, target_3d_traj, w)

        loss_total = loss_3d_traj * 1.

        loss_total.backward()
        nn.utils.clip_grad_norm_(self.model_traj_train.parameters(), max_norm=1)  #
        self.optimizer_T.step()

        epoch_loss_3d_train += target_3d_traj.shape[0] * target_3d_traj.shape[1] * loss_3d_traj.item()
        N += target_3d_traj.shape[0] * target_3d_traj.shape[1]

        return loss_3d_traj.detach(), epoch_loss_3d_train, N

    def train_posenet_realpose(self, tag='_real'):
        """
        _real: dataloader from random projection
        """
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        self.model_pos_train.train()

        # Regular supervised scenario
        self.logger.info(
            'INFO: Train on real pose with dataloader len:{:0>4d}'.format(self.train_generator.num_batches))
        for _, batch_3d, batch_2d in tqdm(self.train_generator.next_epoch()):
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            loss_3d_pos, epoch_loss_3d_train, N = self._train_batch_posenet(inputs_2d, inputs_3d,
                                                                            epoch_loss_3d_train, N)
            # batch-wise log
            self.writer.add_scalar('train_P_batch/{}/loss_3d_pos'.format(tag), loss_3d_pos.item(),
                                   self.summary.train_realpose_iter_num)
            self.summary.summary_train_realpose_iter_num_update()


    def train_trajnet_realpose(self, tag='_real'):
        """
        _real: dataloader from random projection
        """
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        self.model_traj_train.train()

        # Regular supervised scenario
        self.logger.info(
            'INFO: Train on real pose with dataloader len:{:0>4d}'.format(self.train_generator.num_batches))
        for _, batch_3d, batch_2d in tqdm(self.train_generator.next_epoch()):
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            loss_3d_traj, epoch_loss_3d_train, N = self._train_batch_trajnet(inputs_2d, inputs_3d,
                                                                            epoch_loss_3d_train, N)
            # batch-wise log
            self.writer.add_scalar('train_T_batch/{}/loss_3d_traj'.format(tag), loss_3d_traj.item(),
                                   self.summary.train_realtraj_iter_num)
            self.summary.summary_train_realtraj_iter_num_update()

    def _train_dis(self, model_dis, data_real, data_fake, writer_name, fake_data_pool, optimizer):
        """
        """
        optimizer.zero_grad()
        data_real = data_real.detach()
        data_fake = data_fake.detach()
        # store the fake buffer for discriminator training.
        data_fake = Variable(
            torch.Tensor(fake_data_pool(data_fake.cpu().detach().data.numpy()))).to(
            self.device)
        # for 3d part
        real_3d = model_dis(data_real)
        fake_3d = model_dis(data_fake)
        real_label_3d = Variable(torch.ones(real_3d.size())).to(self.device)
        fake_label_3d = Variable(torch.zeros(fake_3d.size())).to(self.device)
        dis_3d_real_loss = self.MSE(real_3d, real_label_3d)
        dis_3d_fake_loss = self.MSE(fake_3d, fake_label_3d)

        # Total discriminators losses
        dis_3d_loss = (dis_3d_real_loss + dis_3d_fake_loss) * 0.5

        # record acc
        d3d_real_acc = get_discriminator_accuracy(real_3d.reshape(-1), real_label_3d.reshape(-1))
        d3d_fake_acc = get_discriminator_accuracy(fake_3d.reshape(-1), fake_label_3d.reshape(-1))

        self.writer.add_scalar(writer_name + '_real_acc', d3d_real_acc, self.summary.train_iter_num)
        self.writer.add_scalar(writer_name + '_fake_acc', d3d_fake_acc, self.summary.train_iter_num)
        self.writer.add_scalar(writer_name + '_dis_loss', dis_3d_loss.item(), self.summary.train_iter_num)

        # Update optimizer
        ###################################################
        dis_3d_loss.backward()
        nn.utils.clip_grad_norm_(model_dis.parameters(), max_norm=1)
        optimizer.step()
        return d3d_real_acc, d3d_fake_acc

    def evaluate_posenet(self, tag='real', valset='s911'):
        """
        evaluate the performance of posenet on 3 kinds of dataset
        """
        start_time = time()
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_pos.load_state_dict(self.model_pos_train.state_dict())
            self.model_pos.eval()

            epoch_p1_3d_valid = 0
            epoch_p2_3d_valid = 0
            N_valid = 0

            test_generator = self.val_generator_dict[valset]
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                # inputs_3d[:, :, 0] = 0
                inputs_3d[:, :, :, :] = inputs_3d[:, :, :, :] - inputs_3d[:, :, :1, :]

                # Predict 3D poses
                predicted_3d_pos = self.model_pos(inputs_2d)

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    predicted_3d_pos[1, :, test_generator.joints_left + test_generator.joints_right] = \
                        predicted_3d_pos[1, :, test_generator.joints_right + test_generator.joints_left]

                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                    inputs_3d = inputs_3d[:1]

                loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_p1_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                N_valid += inputs_3d.shape[0] * inputs_3d.shape[1]

                p2_3d_pos = p_mpjpe(predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                                    , inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1]))
                epoch_p2_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * p2_3d_pos.item()

            # analysis result
            epoch_p1_3d_valid = epoch_p1_3d_valid / N_valid * 1000
            epoch_p2_3d_valid = epoch_p2_3d_valid / N_valid * 1000

        elapsed = (time() - start_time) / 60

        # epoch-wise log.
        self.writer.add_scalar('eval_P_epoch_{}/{}_p1'.format(tag, valset), epoch_p1_3d_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_epoch_{}/{}_p2'.format(tag, valset), epoch_p2_3d_valid, self.summary.epoch)

        return {'p1': epoch_p1_3d_valid}

    def evaluate_posenet_withPCK(self, tag='real', valset='3dhp_flip'):
        """
        evaluate the performance of posenet for 3DHP
        :return:
        """
        start_time = time()
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_pos.load_state_dict(self.model_pos_train.state_dict())
            self.model_pos.eval()

            epoch_p1_3d_valid = 0
            epoch_p2_3d_valid = 0
            epoch_pck_3d_valid = 0
            epoch_auc_3d_valid = 0
            epoch_pck_3dscaled_valid = 0
            epoch_auc_3dscaled_valid = 0
            epoch_pck_3daligned_valid = 0
            epoch_auc_3daligned_valid = 0
            N_valid = 0

            test_generator = self.val_generator_dict[valset]
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                # inputs_3d[:, :, 0] = 0
                inputs_3d[:, :, :, :] = inputs_3d[:, :, :, :] - inputs_3d[:, :, :1, :]

                # Predict 3D poses
                predicted_3d_pos = self.model_pos(inputs_2d)

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    predicted_3d_pos[1, :, test_generator.joints_left + test_generator.joints_right] = \
                        predicted_3d_pos[1, :, test_generator.joints_right + test_generator.joints_left]

                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                    inputs_3d = inputs_3d[:1]

                # to numpy
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                inputs_3d = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                # align a pose result
                predicted_3d_pos_aligned = pose_align(predicted_3d_pos, inputs_3d)
                # align a pose result
                predicted_3d_pos_scaled = pose_scaled(torch.from_numpy(predicted_3d_pos).unsqueeze(0), torch.from_numpy(inputs_3d).unsqueeze(0)).squeeze(0).cpu().numpy()

                # caculate p1 p2 pck auc
                loss_3d_pos = mpjpe(torch.from_numpy(predicted_3d_pos), torch.from_numpy(inputs_3d)).item() * 1000.0
                p2_3d_pos = p_mpjpe(predicted_3d_pos, inputs_3d).item() * 1000.0
                # compute AUC and PCK
                pck = compute_PCK(inputs_3d, predicted_3d_pos)
                auc = compute_AUC(inputs_3d, predicted_3d_pos)

                # compute AUC and PCK after aligned
                pck_aligned = compute_PCK(inputs_3d, predicted_3d_pos_aligned)
                auc_aligned = compute_AUC(inputs_3d, predicted_3d_pos_aligned)

                # compute AUC and PCK after aligned
                pck_scaled = compute_PCK(inputs_3d, predicted_3d_pos_scaled)
                auc_scaled = compute_AUC(inputs_3d, predicted_3d_pos_scaled)

                epoch_p1_3d_valid += inputs_3d.shape[0] * loss_3d_pos
                epoch_p2_3d_valid += inputs_3d.shape[0] * p2_3d_pos
                epoch_pck_3d_valid += inputs_3d.shape[0] * pck
                epoch_auc_3d_valid += inputs_3d.shape[0] * auc

                epoch_pck_3daligned_valid += inputs_3d.shape[0] * pck_aligned
                epoch_auc_3daligned_valid += inputs_3d.shape[0] * auc_aligned

                epoch_pck_3dscaled_valid += inputs_3d.shape[0] * pck_scaled
                epoch_auc_3dscaled_valid += inputs_3d.shape[0] * auc_scaled

                N_valid += inputs_3d.shape[0]

            # analysis result
            epoch_p1_3d_valid = epoch_p1_3d_valid / N_valid
            epoch_p2_3d_valid = epoch_p2_3d_valid / N_valid
            epoch_pck_3d_valid = epoch_pck_3d_valid / N_valid
            epoch_auc_3d_valid = epoch_auc_3d_valid / N_valid

            epoch_pck_3daligned_valid = epoch_pck_3daligned_valid / N_valid
            epoch_auc_3daligned_valid = epoch_auc_3daligned_valid / N_valid

            epoch_pck_3dscaled_valid = epoch_pck_3dscaled_valid / N_valid
            epoch_auc_3dscaled_valid = epoch_auc_3dscaled_valid / N_valid

        elapsed = (time() - start_time) / 60

        # epoch-wise log.
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_p1'.format(tag, valset), epoch_p1_3d_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_p2'.format(tag, valset), epoch_p2_3d_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_pck'.format(tag, valset), epoch_pck_3d_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_auc'.format(tag, valset), epoch_auc_3d_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_pck_aligned'.format(tag, valset), epoch_pck_3daligned_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_auc_aligned'.format(tag, valset), epoch_auc_3daligned_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_pck_scaled'.format(tag, valset), epoch_pck_3dscaled_valid, self.summary.epoch)
        self.writer.add_scalar('eval_P_pck_epoch_{}/{}_auc_scaled'.format(tag, valset), epoch_auc_3dscaled_valid, self.summary.epoch)

        return {
            'p1': epoch_p1_3d_valid,
            'p2': epoch_p2_3d_valid,
        }


    def evaluate_trajnet(self, tag='real', valset='s911'):
        """
        evaluate the performance of posenet
        """
        start_time = time()
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_traj.load_state_dict(self.model_traj_train.state_dict())
            self.model_traj.eval()

            epoch_p1_3d_valid = 0
            N_valid = 0

            # Evaluate on test set
            for cam, batch, batch_2d in self.val_generator_dict[valset].next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                target_3d_traj = inputs_3d[:, :, :1, :] * 1.  # focus on root traj.

                # Predict 3D trajes
                predicted_3d_traj = self.model_traj(inputs_2d)
                loss_3d_traj = mpjpe(predicted_3d_traj, target_3d_traj)
                epoch_p1_3d_valid += target_3d_traj.shape[0] * target_3d_traj.shape[1] * loss_3d_traj.item()
                N_valid += target_3d_traj.shape[0] * target_3d_traj.shape[1]

            # analysis result
            epoch_p1_3d_valid = epoch_p1_3d_valid / N_valid * 1000

        elapsed = (time() - start_time) / 60

        # epoch-wise log.
        self.writer.add_scalar('eval_T_epoch_{}/{}_p1'.format(tag, valset), epoch_p1_3d_valid, self.summary.epoch)

        return {'p1': epoch_p1_3d_valid}

    def _zip_GIFplot_array(self, tensor_lst):
        """
        for plot function pre-preocess
        """
        lst = []
        for item in tensor_lst:
            if item.shape[-1] == 3:  # for 3D case
                lst.append(item.detach().cpu().numpy()[:1])
            elif item.shape[-1] == 2:
                tmp2d = item.detach().cpu().numpy()[:1]
                tmp2d = np.concatenate([tmp2d, np.zeros_like(tmp2d)[..., -1:]], axis=-1)
                lst.append(tmp2d)
            else:
                assert False, 'wrong data get'
        return np.concatenate(lst)

    def random_aug_d2d(self, x):
        r1 = self.args.d2d_random_lb
        r2 = self.args.d2d_random_ub
        random_weight = torch.FloatTensor(x.shape[0], 1, 1, 1).uniform_(r1, r2).to(x.device)
        return x * random_weight

