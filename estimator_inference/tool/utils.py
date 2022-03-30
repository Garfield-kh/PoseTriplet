import os
import time

from common.loss import *

# record time
def update_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()

def mkd(target_dir, get_parent=True):
    # get parent path and create
    if get_parent:
        savedir = os.path.abspath(os.path.join(target_dir, os.pardir))
    else:
        savedir = target_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

def convert_AlphaOpenposeCoco_to_standard16Joint(pose_x):
    """
    pose_x: nx17x2
    https://zhuanlan.zhihu.com/p/367707179
    """
    hip = 0.5 * (pose_x[:, 11] + pose_x[:, 12])
    neck = 0.5 * (pose_x[:, 5] + pose_x[:, 6])
    spine = 0.5 * (neck + hip)

    # head = 0.5 * (pose_x[:, 1] + pose_x[:, 2])

    head_0 = pose_x[:, 0]  # by noise
    head_1 = (neck - hip)*0.5 + neck  # by backbone
    head_2 = 0.5 * (pose_x[:, 1] + pose_x[:, 2])  # by two eye
    head_3 = 0.5 * (pose_x[:, 3] + pose_x[:, 4])  # by two ear
    head = head_0 * 0.1 + head_1 * 0.6 + head_2 * 0.1 + head_3 * 0.2

    combine = np.stack([hip, spine, neck, head])  # 0 1 2 3 ---> 17, 18, 19 ,20
    combine = np.transpose(combine, (1, 0, 2))
    combine = np.concatenate([pose_x, combine], axis=1)
    reorder = [17, 12, 14, 16, 11, 13, 15, 18, 19, 20, 5, 7, 9, 6, 8, 10]
    standart_16joint = combine[:, reorder]
    return standart_16joint

def convert_hhr_to_standard16Joint(pose_x):
    """
    pose_x: nx17x2
    https://zhuanlan.zhihu.com/p/367707179
    """
    re_order = [3, 12, 14, 16, 11, 13, 15, 1, 2, 0, 4, 5, 7, 9, 6, 8, 10]
    standart_17joint = pose_x[:, re_order]
    standart_16joint = np.delete(standart_17joint, 9, axis=1)
    # standart_16joint = np.delete(standart_17joint, 10, axis=1)
    return standart_16joint

def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    # def get_hr_pose():
    #     from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
    #     return hr_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
        # 'hr_pose': get_hr_pose,
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        # return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])

    #   connected son:    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 12, 13,14, 15, 16  #### for 16 joint.

    def joints_right(self):
        # return [1, 2, 3, 9, 10]
        return [1, 2, 3, 13, 14, 15]

import cv2
import moviepy.video.io.ImageSequenceClip

def image_sequence_to_video(img_lst, output_path, frame_rate):
    # frame_size = (500, 500)
    # output_path = 'output_video.mp4'
    # frame_rate = 25
    frame_size = cv2.imread(img_lst[0]).shape[:2]
    print('frame_size: {}'.format(frame_size))
    if not os.path.isfile(output_path):
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(img_lst, fps=frame_rate)
        clip.write_videofile(output_path)
    else:
        print('{} exist already.'.format(output_path))
    return frame_size
