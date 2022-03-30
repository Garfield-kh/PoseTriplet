import ntpath
import os
import shutil

import numpy as np
import torch.utils.data
from tqdm import tqdm

from SPPE.src.main_fast_inference import *
from common.utils import calculate_area
from dataloader import DetectionLoader, DetectionProcessor, DataWriter, Mscoco, VideoLoader
from fn import getTime
from opt import opt
from pPose_nms import write_json

##
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from pPose_nms import pose_nms
from visualization_copy import render_animation, Skeleton, camera_to_world_bynumpy
##

args = opt
args.vis_fast = False # True  # add for speed
args.dataset = 'coco'
args.fast_inference = False
args.save_img = False  # save time and space.
args.sp = True

############################
########## FAST  ###########
args.fast_inference = True
# --conf: Confidence threshold for human detection.
# Lower the value can improve the final accuracy but decrease the speed. Default is 0.1.
args.conf = 0.1
# --nms: Confidence threshold for human detection.
# Increase the value can improve the final accuracy but decrease the speed. Default is 0.6.
args.nms = 0.6
# --inp_dim: The input size of detection network. The inp_dim should be multiple of 32. Default is 608.
args.inp_dim = 32 * 2  # default 608
############################


if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


# def model_load():
#     model = None
#     return model
#
#
# def image_interface(model, image):
#     pass


# def generate_kpts(video_file):
#     final_result, video_name = handle_video(video_file)
#
#     # ============ Changing ++++++++++
#
#     kpts = []
#     no_person = []
#     for i in range(len(final_result)):
#         if not final_result[i]['result']:  # No people
#             no_person.append(i)
#             kpts.append(None)
#             continue
#
#         kpt = max(final_result[i]['result'],
#                   key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']
#
#         kpts.append(kpt.data.numpy())
#
#         for n in no_person:
#             kpts[n] = kpts[-1]
#         no_person.clear()
#
#     for n in no_person:
#         kpts[n] = kpts[-1] if kpts[-1] else kpts[n-1]
#
#     # ============ Changing End ++++++++++
#
#     # name = f'{args.outputpath}/{video_name}_det2D.npz'
#     # npy_folder = os.path.abspath(os.path.join(video_file, os.pardir))
#     # name = video_file.replace('.mp4', '_det2D.npz').replace('source_video', 'det2D')
#     # mkd(name)
#     # kpts = np.array(kpts).astype(np.float32)
#     # print('kpts npz save in ', name)
#     # np.savez_compressed(name, kpts=kpts)  # move to main file for both detectors
#
#     return kpts


# def handle_video(video_file):
#     # =========== common ===============
#     args.video = video_file
#     base_name = os.path.basename(args.video)
#     video_name = base_name[:base_name.rfind('.')]
#     # =========== end common ===============
#
#     # =========== video ===============
#     args.outputpath = f'outputs/alpha_pose_{video_name}'
#     if os.path.exists(args.outputpath):
#         shutil.rmtree(f'{args.outputpath}/vis', ignore_errors=True)
#     else:
#         os.mkdir(args.outputpath)
#     videofile = args.video
#     mode = args.mode
#     if not len(videofile):
#         raise IOError('Error: must contain --video')
#     # Load input video
#     data_loader = VideoLoader(videofile, batchSize=args.detbatch).start()
#     (fourcc, fps, frameSize) = data_loader.videoinfo()
#     print('the video is {} f/s'.format(fps))
#     print('the video frameSize: {}'.format(frameSize))
#     # =========== end video ===============
#     # Load detection loader
#     print('Loading YOLO model..')
#     sys.stdout.flush()
#     det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
#     #  start a thread to read frames from the file video stream
#     det_processor = DetectionProcessor(det_loader).start()
#     # Load pose model
#     pose_dataset = Mscoco()
#     if args.fast_inference:
#         pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
#     else:
#         pose_model = InferenNet(4 * 1 + 1, pose_dataset)
#     pose_model.cuda()
#     pose_model.eval()
#     runtime_profile = {
#         'dt': [],
#         'pt': [],
#         'pn': []
#     }
#     # Data writer
#     save_path = os.path.join(args.outputpath, 'AlphaPose_' + ntpath.basename(video_file).split('.')[0] + '.avi')
#     # writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()
#     writer = DataWriter(args.save_video).start()
#     print('Start pose estimation...')
#     im_names_desc = tqdm(range(data_loader.length()))
#     batchSize = args.posebatch
#
#     final_result_kh = []
#     for i in im_names_desc:
#
#         start_time = getTime()
#         with torch.no_grad():
#             (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
#             if orig_img is None:
#                 print(f'{i}-th image read None: handle_video')
#                 break
#             if boxes is None or boxes.nelement() == 0:
#                 writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
#                 continue
#
#             ckpt_time, det_time = getTime(start_time)
#             runtime_profile['dt'].append(det_time)
#             # Pose Estimation
#
#             datalen = inps.size(0)
#             leftover = 0
#             if datalen % batchSize:
#                 leftover = 1
#             num_batches = datalen // batchSize + leftover
#             hm = []
#             for j in range(num_batches):
#                 inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
#                 hm_j = pose_model(inps_j)
#                 hm.append(hm_j)
#             hm = torch.cat(hm)
#             ckpt_time, pose_time = getTime(ckpt_time)
#             runtime_profile['pt'].append(pose_time)
#
#             hm = hm.cpu().data
#             writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
#
#             ##>>>>>>>
#             boxes, scores, hm_data, pt1, pt2, orig_img, im_name = boxes, scores, hm, pt1, pt2, orig_img, im_name
#             preds_hm, preds_img, preds_scores = getPrediction(
#                 hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
#             result = pose_nms(
#                 boxes, scores, preds_img, preds_scores)
#             result = {
#                 'imgname': im_name,
#                 'result': result
#             }
#             final_result_kh.append(result)
#             ##<<<<<<<<
#
#             ckpt_time, post_time = getTime(ckpt_time)
#             runtime_profile['pn'].append(post_time)
#
#         if args.profile:
#             # TQDM
#             im_names_desc.set_description(
#                 'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
#                     dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
#             )
#     if (args.save_img or args.save_video) and not args.vis_fast:
#         print('===========================> Rendering remaining images in the queue...')
#         print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
#     while writer.running():
#         pass
#     writer.stop()
#     final_result = writer.results()
#     write_json(final_result, args.outputpath)
#
#     return final_result, video_name




class Detector(object):
    def __init__(self, video_file, cam_wh=None):
        self.img_wh = cam_wh # camera resolution
        self.prepare_datastream(video_file)
        self.prepare_model()

    def prepare_datastream(self, video_file):
        # Load input video
        self.data_loader = VideoLoader(video_file, batchSize=args.detbatch).start()
        (fourcc, fps, frameSize) = self.data_loader.videoinfo()
        print('the video is {} f/s'.format(fps))
        print('the video frameSize: {}'.format(frameSize))
        if self.img_wh is None:
            self.img_wh = max(frameSize)
        # =========== end video ===============

    def prepare_model(self):
        # Load detection loader
        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader(self.data_loader, batchSize=args.detbatch).start()
        #  start a thread to read frames from the file video stream
        self.det_processor = DetectionProcessor(det_loader).start()
        # Load pose model
        pose_dataset = Mscoco()
        if args.fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()
        # time measurement
        self.runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

    def get_kpt(self):
        batchSize = args.posebatch
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = self.det_processor.read()
            if orig_img is None:
                assert False, f'image read None: handle_video'
            if boxes is None or boxes.nelement() == 0:
                # continue
                return None
            ckpt_time, det_time = getTime(start_time)
            self.runtime_profile['dt'].append(det_time)

            # Pose Estimation
            datalen = inps.size(0)
            leftover = 0
            if datalen % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            self.runtime_profile['pt'].append(pose_time)

            hm = hm.cpu().data
            boxes, scores, hm_data, pt1, pt2, orig_img, im_name = boxes, scores, hm, pt1, pt2, orig_img, im_name
            preds_hm, preds_img, preds_scores = getPrediction(
                hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            result = pose_nms(
                boxes, scores, preds_img, preds_scores)
            ckpt_time, post_time = getTime(ckpt_time)
            self.runtime_profile['pn'].append(post_time)

            # assume the largest one person
            kpt = max(result, key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']
            keypoint = kpt.data.numpy()

            # post process
            # keypoint = self.convert_AlphaOpenposeCoco_to_standard16Joint(keypoint.copy())  # Nx16x2
            return keypoint

    def post_process(self, pose17j):
        """
        把img space的17 joint转换成 我用的16 joint.
        :return:
        """
        pose16j = self.convert_AlphaOpenposeCoco_to_standard16Joint(pose17j)
        # normlization keypoint
        keypoint_imgnorm = self.normalize_screen_coordinates(pose16j[..., :2], w=self.img_wh, h=self.img_wh)
        return {
            'pose16j': pose16j,
            'keypoint_imgnorm': keypoint_imgnorm,
        }

    def convert_AlphaOpenposeCoco_to_standard16Joint(self, pose_x):
        """
        pose_x: nx17x2
        https://zhuanlan.zhihu.com/p/367707179
        """
        single_pose = False
        if not len(pose_x.shape) == 3:
            single_pose = True
            pose_x = pose_x.reshape(1, 17, 2)

        hip = 0.5 * (pose_x[:, 11] + pose_x[:, 12])
        neck = 0.5 * (pose_x[:, 5] + pose_x[:, 6])
        spine = 0.5 * (neck + hip)

        # head = 0.5 * (pose_x[:, 1] + pose_x[:, 2])

        head_0 = pose_x[:, 0]  # by noise
        head_1 = (neck - hip) * 0.5 + neck  # by backbone
        head_2 = 0.5 * (pose_x[:, 1] + pose_x[:, 2])  # by two eye
        head_3 = 0.5 * (pose_x[:, 3] + pose_x[:, 4])  # by two ear
        head = head_0 * 0.1 + head_1 * 0.6 + head_2 * 0.1 + head_3 * 0.2

        combine = np.stack([hip, spine, neck, head])  # 0 1 2 3 ---> 17, 18, 19 ,20
        combine = np.transpose(combine, (1, 0, 2))
        combine = np.concatenate([pose_x, combine], axis=1)
        reorder = [17, 12, 14, 16, 11, 13, 15, 18, 19, 20, 5, 7, 9, 6, 8, 10]
        standart_16joint = combine[:, reorder]
        if single_pose:
            standart_16joint = standart_16joint[0]
        return standart_16joint

    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / w * 2 - [1, h / w]

    def image_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        # Reverse camera frame normalization
        return (X + [1, h / w]) * w / 2

    def save_video(self, viz_video_path, viz_output_path, keypoints, prediction, camspace=True):
        """
        把拿到的2D keypoint, 3D pose, 保存成video, 用来debug.
        :param viz_video_path: 原始video地址
        :param viz_output_path: 输出结果的video地址.
        :param keypoints: 2D keypoint, # tx16x2 用pixcel地址
        :param prediction: 3D prediction, # tx16x3 看是world space还是camera space
        :return:
        """
        def _postprocess(prediction):
            # # camera rotation
            rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
            prediction_world = camera_to_world_bynumpy(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction_world[:, :, 2] -= np.min(prediction_world[:, :, 2])
            return prediction_world
        # 准备需要可视化的 3D pose 序列
        if camspace:
            prediction = _postprocess(prediction)
        anim_output = {'Reconstruction': prediction}

        args.frame_rate = 25
        args.viz_bitrate = 30000 # bitrate for mp4 videos
        args.viz_limit = -1 # only render first N frames
        args.viz_downsample = 1 # downsample FPS by a factor N
        args.viz_size = 5 # image size
        args.viz_skip = 0 # skip first N frames of input video
        args.video_width, args.video_height = self.data_loader.videoinfo()[2]

        # 可视化生成.
        print('Rendering... save to {}'.format(viz_output_path))
        render_animation(keypoints, anim_output,
                         Skeleton(), args.frame_rate, args.viz_bitrate, np.array(70., dtype=np.float32), viz_output_path,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=viz_video_path, viewport=(args.video_width, args.video_height),
                         input_video_skip=args.viz_skip)



def generate_kpts_byclass(video_file):
    """
    一个样本测试文件, 输入video的地址, 输出kpts.
    :param video_file:
    :return:
    """
    detclass = Detector(video_file)
    kpts=[]
    im_names_desc = range(detclass.data_loader.length())
    for i in im_names_desc:
        kpts.append(detclass.get_kpt())
    print("---------------- finish kpts ............")
    return kpts


# def mkd(target_dir, get_parent=True):
#     # get parent path and create
#     if get_parent:
#         savedir = os.path.abspath(os.path.join(target_dir, os.pardir))
#     else:
#         savedir = target_dir
#     if not os.path.exists(savedir):
#         os.makedirs(savedir, exist_ok=True)


if __name__ == "__main__":
    os.chdir('../..')
    print(os.getcwd())

    # handle_video(img_path='outputs/image/kobe')
    # generate_kpts('outputs/dance.mp4')
