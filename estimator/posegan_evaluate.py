# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import multiprocessing
import os
import pickle
import random
from time import time

import numpy as np
import torch

from bvh_skeleton import humanoid_1205_skeleton
from common.arguments import parse_args
from common.camera2world import camera_to_worldByPCA
from common.loss import *
from common.model import *
from function.gan_utils import pose_seq_bl_reset
from function.utils import mkd
from function.viz import Wrap_plot_seq_gif
from posegan_basementclass import PoseGANBasement

'''
inference file
'''


class PoseGAN(PoseGANBasement):
    def __init__(self, args):
        PoseGANBasement.__init__(self, args)
        # init param
        # self.augment_len = 32
        self.MSE = nn.MSELoss(reduction='mean').to(self.device)
        # prepare data and dataloader
        self.data_preparation()
        self.dataloader_preparation()
        # prepare model
        self.model_preparation()

    def model_preparation(self):
        self._model_preparation_pos()
        self._model_preparation_traj()

    def fit(self, args):
        ###################################
        # Train start here.
        ###################################
        # load pretrain.
        if args.pretrain:
            self.logger.info('Check pretrain model performance.')
            val_rlt = {}
            for val_set_key in self.val_generator_dict:
                self.evaluate_posenet(tag='fake', valset=val_set_key)
                self.evaluate_trajnet(tag='fake', valset=val_set_key)
            # for val_set_key in self.val_generator_dict:
            #     val_rlt[val_set_key] = self.evaluate_posenet(tag='real', valset=val_set_key)
            # self.summary.summary_epoch_update()

            # vis result on s15678
            val_set_key = 's15678'
            # val_set_key = 's15678_flip'
            tag = 'fake'
            self.save_result(valset=val_set_key)
            self.vis_result(tag='fake', valset=val_set_key)


    def evaluate_posenet(self, tag='real', valset='s911'):
        """
        evaluate the performance of posenet on 3 kinds of dataset
        check every clip performance.
        """
        start_time = time()
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_pos.load_state_dict(self.model_pos_train.state_dict())
            self.model_pos.eval()

            epoch_p1_3d_valid = 0
            N_valid = 0

            test_generator = self.val_generator_dict[valset]
            for cam_ex, batch, batch_2d in test_generator.next_epoch():
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

                # batch-wise log.
                self.writer.add_scalar('eval_P_iter_{}/{}_p1'.format(tag, valset), loss_3d_pos.item() * 1000,
                                       self.summary.test_iter_num)
                self.summary.summary_test_iter_num_update()

            # analysis result
            epoch_p1_3d_valid = epoch_p1_3d_valid / N_valid * 1000

        elapsed = (time() - start_time) / 60

        # epoch-wise log.
        self.writer.add_scalar('eval_P_epoch_{}/{}_p1'.format(tag, valset), epoch_p1_3d_valid, self.summary.epoch)

        return

    def evaluate_trajnet(self, tag='real', valset='s911'):
        """
        evaluate the performance of posenet on 3 kinds of dataset
        """
        start_time = time()
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_traj.load_state_dict(self.model_traj_train.state_dict())
            self.model_traj.eval()

            epoch_p1_3d_valid = 0
            N_valid = 0
            self.summary.test_iter_num = 0  # reset here.

            # Evaluate on test set
            for cam, batch, batch_2d in self.val_generator_dict[valset].next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                # else:
                target_3d_traj = inputs_3d[:, :, :1, :] * 1.  # focus on root traj.

                # Predict 3D trajes
                predicted_3d_traj = self.model_traj(inputs_2d)
                loss_3d_traj = mpjpe(predicted_3d_traj, target_3d_traj)
                epoch_p1_3d_valid += target_3d_traj.shape[0] * target_3d_traj.shape[1] * loss_3d_traj.item()
                N_valid += target_3d_traj.shape[0] * target_3d_traj.shape[1]

                # batch-wise log.
                self.writer.add_scalar('eval_T_iter_{}_traj_error/{}'.format(tag, valset), loss_3d_traj.item() * 1000,
                                       self.summary.test_iter_num)

                # check vel.
                max_traj_pred = torch.max(torch.norm(predicted_3d_traj, dim=len(predicted_3d_traj.shape)-1))
                max_traj_gt = torch.max(torch.norm(target_3d_traj, dim=len(target_3d_traj.shape)-1))
                max_traj_error = torch.max(torch.norm(predicted_3d_traj-target_3d_traj, dim=len(target_3d_traj.shape)-1))
                self.writer.add_scalar('eval_T_iter_{}_max_traj_pred/{}'.format(tag, valset),
                                       max_traj_pred.item() * 1000, self.summary.test_iter_num)
                self.writer.add_scalar('eval_T_iter_{}_max_traj_gt/{}'.format(tag, valset),
                                       max_traj_gt.item() * 1000, self.summary.test_iter_num)
                self.writer.add_scalar('eval_T_iter_{}_max_traj_error/{}'.format(tag, valset),
                                       max_traj_error.item() * 1000, self.summary.test_iter_num)

                self.summary.summary_test_iter_num_update()

            # analysis result
            epoch_p1_3d_valid = epoch_p1_3d_valid / N_valid * 1000

        elapsed = (time() - start_time) / 60

        # epoch-wise log.
        self.writer.add_scalar('eval_T_epoch_{}/{}_p1'.format(tag, valset), epoch_p1_3d_valid, self.summary.epoch)

        return {'p1': epoch_p1_3d_valid}

    def vis_result(self, tag='real', valset='s911'):
        """
        evaluate the performance of posenet on 3 kinds of dataset
        check every clip performance.
        """
        start_time = time()
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_pos.load_state_dict(self.model_pos_train.state_dict())  # 这个操作我很喜欢
            self.model_pos.eval()
            self.model_traj.load_state_dict(self.model_traj_train.state_dict())  # 这个操作我很喜欢
            self.model_traj.eval()

            epoch_p1_3d_valid = 0
            N_valid = 0
            batch_num = 0

            # Evaluate on test set
            test_generator = self.val_generator_dict[valset]
            for cam_ex, batch, batch_2d in test_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                cam_ex = torch.from_numpy(cam_ex.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    cam_ex = cam_ex.cuda()
                # inputs_3d[:, :, 0] = 0
                inputs_3d_origin = inputs_3d * 1.  # a copy.
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
                    cam_ex = cam_ex[:1]

                # Predict 3D traj
                predicted_3d_traj = self.model_traj(inputs_2d)
                predicted_3d_traj_withroot = predicted_3d_traj[:1]
                # combine root and pose.
                predicted_3d_pos_withroot = predicted_3d_pos + predicted_3d_traj_withroot
                # convert to the world space.
                # predicted_3d_wpos_withroot = cam2world_sktpos(predicted_3d_pos_withroot)
                # predicted_3d_wpos_withroot = camera_to_worldByTensor(predicted_3d_pos_withroot, cam_ex[..., :4], cam_ex[..., 4:])
                # predicted_3d_wpos_withroot = camera_to_worldByTensor(predicted_3d_pos_withroot, cam_ex[..., :4],
                #                                                      torch.zeros_like(cam_ex[..., 4:]))
                predicted_3d_wpos_withroot = camera_to_worldByPCA(predicted_3d_pos_withroot)

                # visualize result
                if batch_num % 100 == 0 or batch_num in [599]:
                    lables = ['predict_cam3d', 'input_cam3d', 'input_cam2d', 'predict_withroot_cam3d',
                              'predict_withroot_world']

                    clip_len = predicted_3d_pos.shape[1]
                    vis_len = clip_len if clip_len < 1000 else 1000
                    downsample_idx = np.arange(0, vis_len, 10)

                    seqs = self._zip_GIFplot_array([
                        predicted_3d_pos[:, downsample_idx], inputs_3d_origin[:, downsample_idx],
                        inputs_2d[:, downsample_idx], predicted_3d_pos_withroot[:, downsample_idx],
                        predicted_3d_wpos_withroot[:, downsample_idx],
                    ])
                    gif_save_path = os.path.join(args.checkpoint, 'EvaluationGif/{}/epoch{:0>3d}_batch{:0>3d}.gif'.format(valset,
                        self.summary.epoch,  batch_num))
                    self.logger.info('plotting image-->{}'.format(gif_save_path))
                    Wrap_plot_seq_gif(seqs=seqs, labs=lables, save_path=gif_save_path)

                batch_num = batch_num + 1

        return

    def save_result(self, valset):
        """
        evaluate and save the s15678 / s15678_flip
        """
        start_time = time()
        # result_lst = []
        result_all_lst = []
        # End-of-epoch evaluation
        with torch.no_grad():
            self.model_pos.load_state_dict(self.model_pos_train.state_dict())
            self.model_pos.eval()
            self.model_traj.load_state_dict(self.model_traj_train.state_dict())
            self.model_traj.eval()

            # Evaluate on test set
            test_generator = self.val_generator_dict[valset]
            for cam_ex, batch, batch_2d in test_generator.next_epoch():
            # for cam_ex, batch, batch_2d in self.val_generator_dict[valset].next_epoch():
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                cam_ex = torch.from_numpy(cam_ex.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                    cam_ex = cam_ex.cuda()
                # inputs_3d[:, :, 0] = 0
                inputs_3d_origin = inputs_3d * 1.  # a copy.
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
                    cam_ex = cam_ex[:1]

                # Predict 3D traj
                predicted_3d_traj = self.model_traj(inputs_2d)
                predicted_3d_traj_withroot = predicted_3d_traj[:1]
                # combine root and pose.
                predicted_3d_pos_withroot = predicted_3d_pos + predicted_3d_traj_withroot
                # change to world space
                # predicted_3d_wpos_withroot = cam2world_sktpos(predicted_3d_pos_withroot)
                # predicted_3d_wpos_withroot = camera_to_worldByTensor(predicted_3d_pos_withroot, cam_ex[..., :4], cam_ex[..., 4:])
                # predicted_3d_wpos_withroot = camera_to_worldByTensor(predicted_3d_pos_withroot, cam_ex[..., :4],
                #                                                  torch.zeros_like(cam_ex[..., 4:]))
                predicted_3d_wpos_withroot = camera_to_worldByPCA(predicted_3d_pos_withroot, cam_ex[..., :4])

                # fake some qpos data, later will be replaced.
                predicted_3d_qpos = np.random.randn(inputs_3d.shape[1], 59)
                predicted_3d_qpos[:, :7] = np.array([0,0,2,0,0,0,1])

                result_all_lst.append({
                    'cam_ex': cam_ex.detach().cpu().numpy()[0],
                    'inputs_2d': inputs_2d.detach().cpu().numpy()[0, self.pad:-self.pad],
                    'inputs_3d': inputs_3d_origin.detach().cpu().numpy()[0],
                    'predicted_3d': predicted_3d_pos_withroot.detach().cpu().numpy()[0],
                    'predicted_3d_wpos': predicted_3d_wpos_withroot.detach().cpu().numpy()[0],
                    'predicted_3d_qpos': predicted_3d_qpos,
                })

        ##########################################################
        # save result.
        # result_dict = {}
        takes = ['h36m_take_{:0>3d}'.format(i) for i in range(600)]
        # takes = ['h36m_take_{:0>3d}'.format(i) for i in range(60)]
        result_all_dict = {}
        for i, take in enumerate(takes):
            result_all_dict[take] = result_all_lst[i]
        mkd(self.args.traj_save_path)
        with open(self.args.traj_save_path, 'wb') as f:
            pickle.dump(result_all_dict, f)
        # save bvh in multi-process.
        # for i, take in enumerate(takes):
        #     predicted_3d_wpos_withroot = result_all_dict[take]['predicted_3d_wpos']
        #     bvhfileName = self.args.traj_save_path.replace('traj_dict/traj_dict.pkl', 'traj_bvh/'+take+'.bvh')
        #     self.write_standard_bvh(bvhfileName, predicted_3d_wpos_withroot)
        self.write_standard_bvh_multi_process(takes, result_all_dict)
        ##########################################################
        return

    def write_standard_bvh_multi_process(self, takes, result_all_dict):

        def wrap_write_standard_bvh(take):
            predicted_3d_wpos_withroot = np.copy(result_all_dict[take]['predicted_3d_wpos'])
            # reset bl to rl setting
            predicted_3d_wpos_withroot = pose_seq_bl_reset(torch.from_numpy(predicted_3d_wpos_withroot)).numpy()
            # ground_z = np.min(predicted_3d_wpos_withroot[:, :, -1:])
            ground_z = np.min(predicted_3d_wpos_withroot[:, :, -1:], axis=(1,2), keepdims=True)
            predicted_3d_wpos_withroot[:, :, -1:] = predicted_3d_wpos_withroot[:, :, -1:] - ground_z
            bvhfileName = self.args.traj_save_path.replace('traj_dict/traj_dict.pkl', 'traj_bvh/'+take+'.bvh')
            self.write_standard_bvh(bvhfileName, predicted_3d_wpos_withroot)

        # start
        task_lst = takes
        num_threads = args.num_threads

        for ep in range(math.ceil(len(task_lst) / num_threads)):

            p_lst = []
            for i in range(num_threads):
                idx = ep * num_threads + i
                if idx >= len(task_lst):
                    break
                p = multiprocessing.Process(target=wrap_write_standard_bvh, args=(task_lst[idx],))
                p_lst.append(p)

            for p in p_lst:
                p.start()

            for p in p_lst:
                p.join()

            print('complete ep:', ep)
        # end.

    def write_standard_bvh(self, bvhfileName, prediction3dpoint):
        '''
        :param outbvhfilepath:
        :param prediction3dpoint:
        :return:
        '''

        # scale 100 for bvhacker vis.
        for frame in prediction3dpoint:
            for point3d in frame:
                point3d[0] *= 100
                point3d[1] *= 100
                point3d[2] *= 100

        mkd(bvhfileName)
        # 16 joint to 21 joint
        Converter = humanoid_1205_skeleton.SkeletonConverter()
        prediction3dpoint = Converter.convert_to_21joint(prediction3dpoint)
        # save bvh.
        human36m_skeleton = humanoid_1205_skeleton.H36mSkeleton()
        human36m_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)


if __name__ == '__main__':
    args = parse_args()
    # fix random
    random_seed = args.random_seed  # default 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    traj_folder = os.path.abspath(os.path.join(args.traj_save_path, os.pardir))
    args.checkpoint = os.path.join(traj_folder, 'vpose_log')

    mod = PoseGAN(args)
    mod.fit(args)
    mod.writer.close()
