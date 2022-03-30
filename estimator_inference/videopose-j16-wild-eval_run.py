import pickle

from common.arguments import parse_args
from common.camera import camera_to_world, normalize_screen_coordinates, image_coordinates
from common.generators import UnchunkedGenerator
from common.utils import evaluate, add_path
from tool.utils import *
import scipy.signal
import glob

add_path()

"""
inference code for in the wild case.
and save 3D poses for in the wild imitation.
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# for test augmentation
metadata = {'layout_name': 'std', 'num_joints': 16,
            # 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
            'keypoints_symmetry': [[4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]]}


class Visualization(object):
    def __init__(self, ckpt_path):
        self.current_time = time0
        self.ckpt_path = ckpt_path
        self.root_trajectory = None
        self.set_param()
        self.get_video_wh()
        self.get_keypoints()

    def redering(self):
        # poseaug result
        # architecture = '3,3,3,1,1' # add for custom
        architecture = args.architecture
        result = self.get_prediction(architecture, self.ckpt_path)

        anim_output = {
            'result': result,
        }
        self.visalizatoin(anim_output)

    def get_video_wh(self):
        vid = cv2.VideoCapture(args.viz_video_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        args.video_width = int(width)  # cv2 read (height, width)
        args.video_height = int(height)
        self.update_time('prepare video clip')


    def set_param(self):
        dir_name = os.path.dirname(args.viz_video_path)
        basename = os.path.basename(args.viz_video_path)
        self.video_name = basename[:basename.rfind('.')]
        self.viz_output_path = f'{dir_name}/{self.video_name}_{args.detector_2d}.mp4'.replace(
            'source_video', args.architecture.replace(',', '')+'_scale2D_{:0>3d}'.format(int(args.pose2dscale * 10)))
        # prepare folder
        mkd(args.viz_video_path, get_parent=True)
        mkd(self.viz_output_path, get_parent=True)
        # init some property
        keypoints_symmetry = metadata['keypoints_symmetry']
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list([4, 5, 6, 10, 11, 12]), list([1, 2, 3, 13, 14, 15])

    def update_time(self, task):
        time_cost, self.current_time = update_time(self.current_time)
        print('-------------- {} spends {:.2f} seconds'.format(task, time_cost))

    def keypoint_square_padding(self, keypoint):
        """
        square_padding
        the same as take the longer one as width.
        """
        tmp_keypoint = keypoint.copy()
        if args.video_width > args.video_height:  # up down padding
            pad = int((args.video_width - args.video_height)*0.5)
            tmp_keypoint[:, :, 1] =  tmp_keypoint[:, :, 1] + pad
            args.video_height = args.video_width
        elif args.video_width < args.video_height:  # left right padding
            pad = int((args.video_height - args.video_width)*0.5)
            tmp_keypoint[:, :, 0] =  tmp_keypoint[:, :, 0] + pad
            args.video_width = args.video_height
        else:
            print('image are square, no need padding')
        return tmp_keypoint


    def get_keypoints(self):
        # 2D kpts loads or generate
        tmp_npz_path = args.viz_video_path.replace('.mp4', '_det2D.npz').replace('source_video', 'det2D_'+args.detector_2d)
        args.input_npz = tmp_npz_path if os.path.isfile(tmp_npz_path) else None
        if not args.input_npz:
            # get detector for unlabeled video
            detector_2d = get_detector_2d(args.detector_2d)
            assert detector_2d, 'detector_2d should be in ({alpha, hr, open}_pose)'
            # detect keypoints
            self.keypoints = detector_2d(args.viz_video_path)
            # save for next time use
            mkd(tmp_npz_path)
            kpts = np.array(self.keypoints).astype(np.float32)
            print('kpts npz save in ', tmp_npz_path)
            np.savez_compressed(tmp_npz_path, kpts=kpts)
        else:
            # load keypoint
            npz = np.load(args.input_npz)
            self.keypoints = npz['kpts']  # (N, 17, 2) - coco format

        if args.pose2d_smoothing:
            self.keypoints = self.keypoint_smoothing(self.keypoints)

        # convert to standard 16 joint
        if args.detector_2d == 'alpha_pose':  # for coco format -> std 16 j
            self.keypoints = convert_AlphaOpenposeCoco_to_standard16Joint(
                self.keypoints.copy())  # Nx16x2
        self.keypoints_imgunnorm = self.keypoint_square_padding(self.keypoints)
        # normlization keypoints
        self.keypoints_imgnorm = normalize_screen_coordinates(self.keypoints_imgunnorm[..., :2], w=args.video_width,
                                                              h=args.video_height)
        self.update_time('load keypoint')
        # analysis scale
        self.keypoints_imgnorm = self.keypoints_imgnorm * args.pose2dscale

    def keypoint_smoothing(self, keypoints):
        x = keypoints.copy()
        window_length = 5
        polyorder = 2
        out = scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
        return out

    def get_prediction(self, architecture, ckpt_path):
        model_pos = self._get_model(architecture, ckpt_path)
        data_loader = self._get_dataloader(model_pos)
        prediction = self._evaluate(model_pos, data_loader)

        if args.add_trajectory:
            model_traj = self._get_modelTraj(architecture, ckpt_path)
            data_loader = self._get_dataloaderTraj(model_traj)
            self.root_trajectory = self._evaluate(model_traj, data_loader)
        return prediction

    def _get_model(self, architecture, ckpt_path):
        from common.model import TemporalModel
        filter_widths = [int(x) for x in architecture.split(',')]
        model_pos = TemporalModel(16, 2, 16, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                  channels=args.channels,
                                  dense=args.dense).cuda()
        # load trained model
        print('Loading checkpoint', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'])
        self.update_time('load 3D model')
        return model_pos

    def _get_modelTraj(self, architecture, ckpt_path):
        from common.model import TemporalModel
        filter_widths = [int(x) for x in architecture.split(',')]
        model_traj = TemporalModel(16, 2, 1, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                  channels=args.channels,
                                  dense=args.dense).cuda()

        # load trained model
        print('Loading checkpoint', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model_traj.load_state_dict(checkpoint['model_traj'])
        self.update_time('load 3D Traj model')
        return model_traj

    def _get_dataloader(self, model_pos):
        #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
        receptive_field = model_pos.receptive_field()
        pad = (receptive_field - 1) // 2  # Padding on each side
        causal_shift = 0
        data_loader = UnchunkedGenerator(None, None, [self.keypoints_imgnorm],
                                         pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                         kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left,
                                         joints_right=self.joints_right)
        return data_loader

    def _get_dataloaderTraj(self, model_pos):
        #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
        receptive_field = model_pos.receptive_field()
        pad = (receptive_field - 1) // 2  # Padding on each side
        causal_shift = 0
        data_loader = UnchunkedGenerator(None, None, [self.keypoints_imgnorm],
                                         pad=pad, causal_shift=causal_shift, augment=False,
                                         kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left,
                                         joints_right=self.joints_right)
        return data_loader

    def _evaluate(self, model_pos, data_loader):
        # get result
        prediction = evaluate(data_loader, model_pos, return_predictions=True,
                              joints_leftright=(self.joints_left, self.joints_right))

        self.update_time('generate reconstruction 3D data')
        return prediction

    def _postprocess(self, prediction):
        if args.add_trajectory:
            # add root trajectory
            prediction -= prediction[:, :1, :]
            prediction += self.root_trajectory
        # # camera rotation
        rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
        prediction_world = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction_world[:, :, 2] -= np.min(prediction_world[:, :, 2])
        return prediction_world

    def visalizatoin(self, anim_output):
        from common.visualization import render_animation
        # anim_output = {'Reconstruction': prediction}
        self.save_3d_prediction(anim_output)

        for tmp_key in anim_output:
            anim_output[tmp_key] = self._postprocess(anim_output[tmp_key])

        if args.pure_background:
            viz_video_path = None
        else:
            viz_video_path = args.viz_video_path

        print('Rendering... save to {}'.format(self.viz_output_path))
        render_animation(self.keypoints, anim_output,
                         Skeleton(), args.frame_rate, args.viz_bitrate, np.array(70., dtype=np.float32), self.viz_output_path,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=viz_video_path, viewport=(args.video_width, args.video_height),
                         input_video_skip=args.viz_skip)
        self.update_time('render animation')

    def save_3d_prediction(self, anim_output):
        tmp_anim_output = {}
        for tmp_key in anim_output:
            prediction =  anim_output[tmp_key] * 1.
            if args.add_trajectory:
                # add root trajectory
                prediction -= prediction[:, :1, :]
                prediction += self.root_trajectory
            tmp_anim_output[tmp_key] = prediction * 1.
        # save 3D joint points
        tmp_pkl_path = args.viz_video_path.replace('.mp4', '_pred3D.pkl').replace('source_video', 'pred3D_pose') # rename to save
        mkd(tmp_pkl_path)
        with open(tmp_pkl_path, 'wb') as handle:
            pickle.dump(tmp_anim_output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    time0 = update_time()
    args = parse_args()

    # model 2d detection detail
    args.detector_2d = 'alpha_pose'

    # redering detail
    args.pure_background = False  # False/True
    args.add_trajectory = True #False
    # args.viz_limit = 200

    ###########################
    # model 2d-3d detail
    ###########################
    args.architecture = '3,3,3'  # model arch
    # args.architecture = '3,1,3,1,3'  # model arch
    args.test_time_augmentation = True  # False
    args.pose2d_smoothing = True

    ckpt_path = './checkpoint/ckpt_ep_045.bin'

    # ================================================================================================
    # seletcted clip run
    # ================================================================================================
    for args.architecture in ['3,3,3']:
        # for args.pose2dscale in [0.5, 0.7, 0.85, 1]:
        for args.pose2dscale in [1]:

            args.eval_data = 'bilibili-clip'
            eval_video_list = glob.glob('./wild_eval/source_video/{}/*.mp4'.format(args.eval_data))
            for path_name in eval_video_list[:1]:
                args.viz_video_path = path_name
                args.frame_rate = 30
                Vis = Visualization(ckpt_path)
                Vis.redering()
