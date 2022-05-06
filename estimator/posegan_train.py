# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
from time import time

import numpy as np
import torch
from torch.autograd import Variable

from common.arguments import parse_args
from common.camera import *
from common.model import *
from function.gan_utils import pose_seq_bl_aug_batch
from function.utils import set_grad, get_discriminator_accuracy
from function.viz import Wrap_plot_seq_gif
from posegan_basementclass import PoseGANBasement
from progress.bar import Bar

'''
training file
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
        self.s911_detect2d_dataloader_preparation()
        # prepare model and optimizer
        self.model_preparation()

    def fit(self, args):
        ###################################
        # Train start here.
        ###################################
        # load pretrain if used.
        if args.pretrain:
            self.logger.info('Check pretrain model performance.')
            val_rlt = {}
            for val_set_key in self.val_generator_dict:
                self.evaluate_posenet(tag='fake', valset=val_set_key)
                self.evaluate_trajnet(tag='fake', valset=val_set_key)
            self.evaluate_posenet_withPCK(tag='fake', valset='3dhp_flip')
            for val_set_key in self.val_generator_dict:
                val_rlt[val_set_key] = self.evaluate_posenet(tag='real', valset=val_set_key)
            self.evaluate_posenet_withPCK(tag='real', valset='3dhp_flip')
            self.summary.summary_epoch_update()

        for epoch in range(args.epochs):
            epoch_start_time = time()

            self.train_posegan()

            if self.summary.epoch > args.P_start_ep:  # start record the performance when P training start.
                val_rlt = {}

                for val_set_key in self.val_generator_dict:
                    val_rlt[val_set_key] = self.evaluate_posenet(tag='fake', valset=val_set_key)
                    self.evaluate_trajnet(tag='fake', valset=val_set_key)
                self.evaluate_posenet_withPCK(tag='fake', valset='3dhp_flip')

                if args.add_random_cam:
                    self.update_fixedfake_train_generator()
                    self.train_posenet_realpose()
                    self.train_trajnet_realpose()
                    for val_set_key in self.val_generator_dict:
                        val_rlt[val_set_key] = self.evaluate_posenet(tag='real', valset=val_set_key)
                    self.evaluate_posenet_withPCK(tag='real', valset='3dhp_flip')  # 单独开一个

                # log
                self.logging(val_rlt, epoch_start_time)

            # udpate per epoch
            self.lr_scheduler_P.step()
            self.lr_scheduler_Gcam.step()
            self.lr_scheduler_Dcam.step()
            self.summary.summary_epoch_update()

    def model_preparation(self):
        self._model_preparation_pos()
        self._model_preparation_traj()
        self._model_preparation_Gcam()

    def train_posenet_fakepose_camed(self, cam_rlt_dict, tag='_fake'):
        epoch_loss_3d_train = 0
        N = 0
        self.model_pos_train.train()
        # prepare fake batch
        pose3D_camed = cam_rlt_dict['pose3D_camed'].detach()
        pose2D_camed = cam_rlt_dict['pose2D_camed'].detach()

        inputs_2d = pose2D_camed.detach()
        inputs_3d = pose3D_camed.detach()[:, 0 + self.pad:0 + self.pad + 1]
        # now get fake data ready for train.
        loss_3d_pos, epoch_loss_3d_train, N = self._train_batch_posenet(inputs_2d.detach(), inputs_3d.detach(),
                                                                        epoch_loss_3d_train, N)
        # batch-wise log
        self.writer.add_scalar('train_P_batch/{}/loss_3d_pos'.format(tag), loss_3d_pos.item(),
                               self.summary.train_fakepose_iter_num)
        self.summary.summary_train_fakepose_iter_num_update()


    def train_trajnet_fakepose_camed(self, cam_rlt_dict, tag='_fake'):
        epoch_loss_3d_train = 0
        N = 0
        self.model_traj_train.train()
        # prepare fake batch
        pose3D_camed = cam_rlt_dict['pose3D_camed'].detach()
        pose2D_camed = cam_rlt_dict['pose2D_camed'].detach()

        inputs_2d = pose2D_camed.detach()
        inputs_3d = pose3D_camed.detach()[:, 0 + self.pad:0 + self.pad + 1]
        # now get fake data ready for train.
        loss_3d_traj, epoch_loss_3d_train, N = self._train_batch_trajnet(inputs_2d.detach(), inputs_3d.detach(),
                                                                        epoch_loss_3d_train, N)
        # batch-wise log
        self.writer.add_scalar('train_T_batch/{}/loss_3d_traj'.format(tag), loss_3d_traj.item(),
                               self.summary.train_faketraj_iter_num)
        self.summary.summary_train_faketraj_iter_num_update()

    def adv_loss(self, model_dis, data_real, data_fake, writer_name):
        # Adversarial losses for 3D squence
        real_3d = model_dis(data_real)
        fake_3d = model_dis(data_fake)

        real_label_3d = Variable(torch.ones(real_3d.size())).to(self.device)
        fake_label_3d = Variable(torch.zeros(fake_3d.size())).to(self.device)

        adv_3d_loss = self.MSE(real_3d, fake_3d)
        # adv_3d_real_loss = self.MSE(real_3d, fake_label_3d)
        # adv_3d_fake_loss = self.MSE(fake_3d, real_label_3d)
        # # Total discriminators losses
        # adv_3d_loss = (adv_3d_real_loss + adv_3d_fake_loss) * 0.5

        # monitor training process
        ###################################################
        real_acc = get_discriminator_accuracy(real_3d.reshape(-1), real_label_3d.reshape(-1))
        fake_acc = get_discriminator_accuracy(fake_3d.reshape(-1), fake_label_3d.reshape(-1))
        self.writer.add_scalar(writer_name + '_real_acc', real_acc, self.summary.train_iter_num)
        self.writer.add_scalar(writer_name + '_fake_acc', fake_acc, self.summary.train_iter_num)
        self.writer.add_scalar(writer_name + '_adv_loss', adv_3d_loss.item(), self.summary.train_iter_num)
        return adv_3d_loss

    def train_posegan(self):
        """
        """
        start_time = time()
        batch_num = 0

        self.model_Gcam.train()
        self.model_Dcam.train()

        bar = Bar('Train pose gan', max=self.aug_generator.num_batches)
        for _, _, batch_3d, batch_2d, _ in self.aug_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))  # b x t x j x 3
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))  # b x t x j x 2


            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            # random bl augment
            inputs_3d = pose_seq_bl_aug_batch(inputs_3d)
            inputs_3dworld_origin = cam2world_sktpos(inputs_3d)

            # train gan
            ##################################################
            #######      Train Generator     #################
            ##################################################
            set_grad([self.model_Gcam], True)
            set_grad([self.model_Dcam], False)
            self.optimizer_Gcam.zero_grad()

            ###################################################
            reset_root = inputs_3d[:, :1, :1, :] * 1.0
            pose_recoverd_uncamed = inputs_3d * 1. - reset_root

            cam_rlt_dict = self.model_Gcam(pose_recoverd_uncamed)
            pose2D_camed = cam_rlt_dict['pose2D_camed']
            adv_cam_loss = self.adv_loss(self.model_Dcam, inputs_2d, pose2D_camed,
                                         writer_name='train_G_iter_acc/gcam')
            # Update generators
            ###################################################
            adv_cam_loss.backward()
            nn.utils.clip_grad_norm_(self.model_Gcam.parameters(), max_norm=1)
            self.optimizer_Gcam.step()

            ################################################
            #######      Train PoseNet     #################
            ################################################
            if self.summary.epoch > args.P_start_ep:
                self.train_posenet_fakepose_camed(cam_rlt_dict)
                self.train_trajnet_fakepose_camed(cam_rlt_dict)

            ##################################################
            #######      Train Discriminator     #############
            ##################################################
            d3d_real_acc, d3d_fake_acc = 0, 0
            if self.summary.train_iter_num % args.df == 0:
                set_grad([self.model_Gcam], False)
                set_grad([self.model_Dcam], True)

                # train Dcam
                d3d_real_acc, d3d_fake_acc = self._train_dis(model_dis=self.model_Dcam,
                                                             data_real=self.random_aug_d2d(inputs_2d),
                                                             data_fake=pose2D_camed,
                                                             writer_name='train_D_iter_acc/dcam',
                                                             fake_data_pool=self.fake_cam_sample,
                                                             optimizer=self.optimizer_Dcam)
            # visualize result
            if self.summary.train_iter_num % 5000 == 0:
                lables = ['input_world', 'input_cam3d', 'input_cam2d', 'pose_recoverd_uncamed_cam3d', 'RT_cam3d', 'RT_cam2d']

                seqs = self._zip_GIFplot_array([
                    inputs_3dworld_origin, inputs_3d, inputs_2d,
                    pose_recoverd_uncamed, cam_rlt_dict['pose3D_camed'], cam_rlt_dict['pose2D_camed']
                ])
                gif_save_path = os.path.join(args.checkpoint, 'trainingGif/epoch{:0>3d}_batch{:0>3d}.gif'.format(
                    self.summary.epoch, batch_num))
                self.logger.info('plotting image-->{}'.format(gif_save_path))
                Wrap_plot_seq_gif(seqs=seqs, labs=lables, save_path=gif_save_path)

            # update writer iter num
            self.summary.summary_train_iter_num_update()

            bar.suffix = '(epoch:{epoch}) | ({batch}/{size}) | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                         '| d3d_real_acc: {d3d_real_acc: .4f} | d3d_fake_acc: {d3d_fake_acc: .4f} ' \
                .format(epoch=self.summary.epoch, batch=batch_num, size=self.aug_generator.num_batches,
                        bt=(time() - start_time) / (batch_num + 1), ttl=bar.elapsed_td, eta=bar.eta_td,
                        d3d_real_acc=d3d_real_acc, d3d_fake_acc=d3d_fake_acc)
            bar.next()
            batch_num = batch_num + 1



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

    mod = PoseGAN(args)
    mod.fit(args)
    mod.writer.close()
