import numpy as np
import os
import yaml
import math
from utils import get_qvel_fd, de_heading


class Dataset:
    """
    get meta-data
    get optic flow // 2d pose
    get gt-trajectory -> cacluate velocity and do normalization
    prepare sampling for training.
    """
    def __init__(self, meta_id, mode, fr_num, iter_method='iter', shuffle=False, overlap=0, num_sample=20000, args=None,
                 mocap_folder = '.'):
        self.meta_id = meta_id
        self.mode = mode
        self.fr_num = fr_num
        self.iter_method = iter_method
        self.shuffle = shuffle
        self.overlap = overlap
        self.num_sample = num_sample
        self.base_folder = os.path.join(mocap_folder, 'datasets')
        # self.of_folder = os.path.join(self.base_folder, 'pose2d')  #of: optic flow.
        # self.traj_folder = os.path.join(self.base_folder, 'traj')
        self.of_folder = os.path.join(self.base_folder, 'pose2d' if not args else args.of_folder)
        # self.traj_folder = os.path.join(self.base_folder, 'traj' if not args else args.traj_folder)
        # self.traj_dict_path = os.path.join(self.traj_folder, 'traj_dict_t0.pkl' if not args else args.traj_dict_path)
        self.traj_dict_path = os.path.join(mocap_folder, 'datasets/traj_dict/traj_dict.pkl')
        # meta_file = '%s/meta/%s.yml' % (self.base_folder, meta_id)
        meta_file = 'datasets/meta/%s.yml' % (meta_id)
        self.meta = yaml.safe_load(open(meta_file, 'r'))
        self.no_traj = self.meta.get('no_traj', False)
        # self.msync = self.meta['video_mocap_sync']
        self.dt = 1 / self.meta['capture']['fps']
        # get take names
        if mode == 'all':
            self.takes = self.meta['train'] + self.meta['test']
        else:
            self.takes = self.meta[mode]
        # preload trajectories
        if self.no_traj:
            self.trajs = None
            self.orig_trajs = None
            self.norm_trajs = None
        else:
            self.trajs = []
            self.trajs_len = {}
            self.orig_trajs = []
            # self.orig_2dpose = []
            traj_dict = np.load(self.traj_dict_path, allow_pickle=True)
            for i, take in enumerate(self.takes):
                # traj_file = '%s/%s_traj.p' % (self.traj_folder, take)
                # orig_traj = np.load(traj_file, allow_pickle=True)
                orig_traj = traj_dict[take]['predicted_3d_qpos']  # follow video3Dqpos.
                # orig_2dpose = traj_dict[take]['inputs_2d']
                # remove noisy hand pose
                # orig_traj[:, 32:35] = 0.0
                # orig_traj[:, 42:45] = 0.0
                traj_pos = self.get_traj_pos(orig_traj)
                traj_vel = self.get_traj_vel(orig_traj)
                traj = np.hstack((traj_pos, traj_vel))
                self.trajs.append(traj)
                self.orig_trajs.append(orig_traj)
                # self.orig_2dpose.append(orig_2dpose)
                self.trajs_len[take] = orig_traj.shape[0]
            if mode == 'train':
                all_traj = np.vstack(self.trajs)
                self.mean = np.mean(all_traj, axis=0)
                self.std = np.std(all_traj, axis=0)
                self.norm_trajs = self.normalize_traj()
            else:
                self.mean, self.std, self.norm_trajs = None, None, None
            self.traj_dim = self.trajs[0].shape[1]
        # iterator specific
        self.sample_count = None
        self.take_indices = None
        self.cur_ind = None
        self.cur_tid = None
        self.cur_fr = None
        self.fr_lb = None
        self.fr_ub = None
        self.im_offset = None

        self.set_msync()  #.
        # get dataset len
        self.len = np.sum([self.msync[x][2] - self.msync[x][1] for x in self.takes])


    def set_msync(self):
        self.msync = {}
        for take in self.trajs_len:
            self.msync[take] = [0, 0, self.trajs_len[take]]

    def __iter__(self):
        if self.iter_method == 'sample':
            self.sample_count = 0
        elif self.iter_method == 'iter':
            self.cur_ind = -1
            self.take_indices = np.arange(len(self.takes))
            if self.shuffle:
                np.random.shuffle(self.take_indices)
            self.__next_take()
        return self

    def __next_take(self):
        self.cur_ind = self.cur_ind + 1
        if self.cur_ind < len(self.take_indices):
            self.cur_tid = self.take_indices[self.cur_ind]  # tid: take_indices
            self.im_offset, self.fr_lb, self.fr_ub = self.msync[self.takes[self.cur_tid]]
            self.cur_fr = self.fr_lb

    def __next__(self):
        if self.iter_method == 'sample':
            if self.sample_count >= self.num_sample:
                raise StopIteration
            self.sample_count += self.fr_num - self.overlap
            return self.sample()
        elif self.iter_method == 'iter':
            if self.cur_ind >= len(self.takes):
                raise StopIteration
            fr_start = self.cur_fr
            fr_end = self.cur_fr + self.fr_num if self.cur_fr + self.fr_num + 30 < self.fr_ub else self.fr_ub
            # print(self.cur_ind, self.cur_tid, fr_start, fr_end)
            of = self.load_of(self.cur_tid, fr_start + self.im_offset, fr_end + self.im_offset)
            if self.no_traj:
                norm_traj, orig_traj = None, None
            else:
                norm_traj = self.norm_trajs[self.cur_tid][fr_start: fr_end]
                orig_traj = self.orig_trajs[self.cur_tid][fr_start: fr_end]
            self.cur_fr = fr_end - self.overlap
            if fr_end == self.fr_ub:
                self.__next_take()
            return of, norm_traj, orig_traj

    def get_traj_pos(self, orig_traj):
        traj_pos = orig_traj[:, 2:].copy()
        for i in range(traj_pos.shape[0]):
            traj_pos[i, 1:5] = de_heading(traj_pos[i, 1:5])
        return traj_pos

    def get_traj_vel(self, orig_traj):
        traj_vel = []
        for i in range(orig_traj.shape[0] - 1):
            vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], self.dt, 'heading')
            traj_vel.append(vel)
        traj_vel.append(traj_vel[-1].copy())
        traj_vel = np.vstack(traj_vel)
        return traj_vel

    def set_mean_std(self, mean, std):
        self.mean, self.std = mean, std
        if not self.no_traj:
            self.norm_trajs = self.normalize_traj()

    def normalize_traj(self):
        """
        :return:
        """
        assert False, 'normalize not in use'
        norm_trajs = []
        for traj in self.trajs:
            norm_traj = (traj - self.mean[None, :]) / (self.std[None, :] + 1e-8)
            norm_trajs.append(norm_traj)
        return norm_trajs

    def sample(self):
        take_ind = np.random.randint(len(self.takes))
        im_offset, fr_lb, fr_ub = self.msync[self.takes[take_ind]]
        fr_start = np.random.randint(fr_lb, fr_ub - self.fr_num)
        fr_end = fr_start + self.fr_num
        of = self.load_of(take_ind, fr_start + im_offset, fr_end + im_offset)
        if self.no_traj:
            norm_traj, orig_traj = None, None
        else:
            norm_traj = self.norm_trajs[take_ind][fr_start: fr_end]
            orig_traj = self.orig_trajs[take_ind][fr_start: fr_end]
        return of, norm_traj, orig_traj

    def load_of(self, take_ind, start, end):
        """
        """
        take_folder = '%s/%s' % (self.of_folder, self.takes[take_ind])
        # of = []
        # for i in range(start, end):
        #     of_file = '%s/%05d.npy' % (take_folder, i)
        #     of_i = np.load(of_file, allow_pickle=True)
        #     of.append(of_i)
        # of = np.stack(of)
        of_file = '%s/pose2d.npy' % (take_folder)
        of_npy = np.load(of_file, allow_pickle=True)
        return of_npy[start:end]




