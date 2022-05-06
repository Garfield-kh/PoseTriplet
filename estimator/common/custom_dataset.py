# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
from common.h36m_dataset import h36m_skeleton
       

custom_camera_params = {
    'id': None,
    'res_w': None, # Pulled from metadata
    'res_h': None, # Pulled from metadata
    
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70, # Only used for visualization
    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}

class CustomDataset(MocapDataset):
    def __init__(self, detections_path, remove_static_joints=True):
        super().__init__(fps=None, skeleton=h36m_skeleton)        
        
        # Load serialized dataset
        data = np.load(detections_path, allow_pickle=True)
        resolutions = data['metadata'].item()['video_metadata']
        
        self._cameras = {}
        self._data = {}
        for video_name, res in resolutions.items():
            cam = {}
            cam.update(custom_camera_params)
            cam['orientation'] = np.array(cam['orientation'], dtype='float32')
            cam['translation'] = np.array(cam['translation'], dtype='float32')
            cam['translation'] = cam['translation']/1000 # mm to meters
            
            cam['id'] = video_name
            cam['res_w'] = res['w']
            cam['res_h'] = res['h']
            
            self._cameras[video_name] = [cam]
        
            self._data[video_name] = {
                'custom': {
                    'cameras': cam
                }
            }
                
        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8
            
    def supports_semi_supervised(self):
        return False

##############################
### for rnn fake data
##############################

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import glob

class MyFakeDataset(Dataset):
    def __init__(self, folder, rf=81, random_t=True):

        self.nplist = glob.glob(folder)
        # inputs_2d_fake = []
        # inputs_3dcam_fake = []
        # cam_in = []
        # cam_ex = []
        # print('start load fake data')
        # for item in tqdm(nplist):
        #     tmp = np.load(item)
        #     inputs_2d_fake.append(tmp['inputs_2d_fake'])
        #     inputs_3dcam_fake.append(tmp['inputs_3dcam_fake'])
        #     cam_in.append(tmp['cam_in'])
        #     cam_ex.append(tmp['cam_ex'])
        #
        # self.inputs_2d_fake = np.concatenate(inputs_2d_fake, axis=0)
        # self.inputs_3dcam_fake = np.concatenate(inputs_3dcam_fake, axis=0)
        # self.cam_in = np.concatenate(cam_in, axis=0)
        # self.cam_ex = np.concatenate(cam_ex, axis=0)
        # print('finish load fake data')
        self.rf = rf
        self.random_t = random_t

    def __getitem__(self, index):
        path = self.nplist[index]
        tmp = np.load(path)

        if self.random_t:
            start_t = int(np.random.randint(0, tmp['inputs_2d_fake'].shape[1] - self.rf, size=1))
        else:
            start_t = int(0)
        end_t = int(start_t + self.rf)
        inputs_2d_fake = tmp['inputs_2d_fake'][:,start_t:end_t]
        inputs_3dcam_fake = tmp['inputs_3dcam_fake'][:,start_t:end_t]
        cam_in = tmp['cam_in'][:,start_t:end_t]
        cam_ex = tmp['cam_ex'][:,start_t:end_t]

        return cam_in, cam_ex, inputs_3dcam_fake, inputs_2d_fake

    def __len__(self):
        return len(self.nplist)

##############################
### for cnn fake data
##############################
class MyFakeDataset_v1(Dataset):
    def __init__(self, folder, rf=81, random_t=True):

        self.nplist = glob.glob(folder)
        print('load fake data: ', len(self.nplist))

    def __getitem__(self, index):
        path = self.nplist[index]
        tmp = np.load(path)
        fake3d_world = np.delete(tmp.squeeze(), 10, axis=0)/1000.0
        fake3d_world = np.transpose(fake3d_world, (1, 0, 2))
        return fake3d_world

    def __len__(self):
        return len(self.nplist)
