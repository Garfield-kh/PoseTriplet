# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision') # semi-sup # not in use
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-n', '--note', default='debug', type=str,
                        help='additional name on checkpoint directory')
    parser.add_argument('-c', '--checkpoint', default='/Checkpoint/gongkehong/June2021/vpos', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=10, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--expert_dict_path', default=None, type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--extra_expert_dict_path', default=None, type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--traj_save_path', default='xxx/traj_dict.pkl', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')
    parser.add_argument('--num-threads', type=int, default=32)

    # Model arguments
    parser.add_argument('--posenet_choice', default='videopose', type=str, metavar='NAME', help='videopose/stgcn/poseformer')
    parser.add_argument('--poseformer_cfg', default='a', type=str, metavar='NAME', help='abcde')
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')  # 这个改了会把pose切段,不好.
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-yes-da', '--yes-data-augmentation', dest='data_augmentation', action='store_true',
                        help='disable train-time flipping')  # not in use, save time
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=1024, type=int, metavar='N', help='number of channels in convolution layers')

    # Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=2, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true', help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true', help='disable optimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true', help='use only linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true', help='disable projection for semi-supervised setting')

    # Experimental
    parser.add_argument('-lrgcam', '--lrgcam', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrdcam', '--lrdcam', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--P_start_ep', default=-1, type=int, metavar='FACTOR', help='warm up before train posenet')
    parser.add_argument('--random_seed', default=0, type=int, metavar='FACTOR', help='random')
    parser.add_argument('--pretrain', default=False, type=lambda x: (str(x).lower() == 'true'), help='...')
    parser.add_argument('--df', default=3, type=int, metavar='FACTOR', help='update frequency for discriminator')
    parser.add_argument('-d-arc', '--Darchitecture', default='3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('-d-ch', '--Dchannels', default=64, type=int, metavar='N', help='number of channels in convolution layers')
    parser.add_argument('--Ddense', action='store_true', help='use dense convolutions instead of dilated convolutions')

    # param for Gcam - camera - default free
    parser.add_argument('--cam_r_range', default=0.5, type=float, help='param for Gcam')
    parser.add_argument('--cam_t_range', default=3, type=float, help='param for Gcam')  # range large, to
    parser.add_argument('--gcam_choice', default='gcam_v0', type=str, metavar='NAME', help='gcam_choice: v0')
    parser.add_argument('--dcam_choice', default='dcam_pa1', type=str, metavar='NAME', help='d3d_choice: pa1')
    parser.add_argument('-dcam-arc', '--Dcamarchitecture', default='64,64', type=str, metavar='LAYERS', help='filter ch separated by comma')
    parser.add_argument('--dcam_ks', default=3, type=int, metavar='N', help='kernel size in dcam')

    parser.add_argument('--dcam_extra_target', default='None', type=str, metavar='NAME', help='dcam_extra_target: 3dhp/3dpw-test')

    parser.add_argument('--d2d_random_lb', default=0.5, type=float, help='param for Gcam')  # range large, to
    parser.add_argument('--d2d_random_ub', default=1.4, type=float, help='param for Gcam')  # range large, to

    # param for random camera rt
    parser.add_argument('--add_random_cam', default=True, type=lambda x: (str(x).lower() == 'true'), help='...')
    parser.add_argument('--rpx_min', default=3.5, type=float, help='param for RandomCam')
    parser.add_argument('--rpx_max', default=6.0, type=float, help='param for RandomCam')
    # parser.add_argument('--rpy_min', default=0, type=float, help='param for RandomCam')
    # parser.add_argument('--rpy_max', default=0, type=float, help='param for RandomCam')
    parser.add_argument('--rpz_min', default=1.4, type=float, help='param for RandomCam')
    parser.add_argument('--rpz_max', default=1.7, type=float, help='param for RandomCam')
    parser.add_argument('--rex_min', default=-115, type=float, help='param for RandomCam')
    parser.add_argument('--rex_max', default=-95, type=float, help='param for RandomCam')
    parser.add_argument('--rey_min', default=-5, type=float, help='param for RandomCam')
    parser.add_argument('--rey_max', default=5, type=float, help='param for RandomCam')
    parser.add_argument('--rez_min', default=80, type=float, help='param for RandomCam')
    parser.add_argument('--rez_max', default=100, type=float, help='param for RandomCam')

    # Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')

    # remove this factor during experiment.
    # parser.set_defaults(bone_length_term=True)
    # parser.set_defaults(data_augmentation=True)
    # parser.set_defaults(test_time_augmentation=True)
    
    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
        
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args