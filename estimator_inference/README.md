# Video to Pose3D (16-joints-setting)

> Predict 3d human pose with root trajectory from in-the-wild video

![alt text](../assets/wild-eval/3x3wildeval.gif)

## Prerequisite

1. Environment
   - Linux system
   - Python > 3.6 distribution
2. Dependencies
   - **Packages**
      - Pytorch = 1.6.0 (Our implementation)
      - Please following the instruction in [video-to-pose3d](https://github.com/zh-plus/video-to-pose3D) for the Packages.
   - **2D Joint detectors**
     - Alphapose (Recommended)
       - Please following the instruction in [video-to-pose3d](https://github.com/zh-plus/video-to-pose3D) to download the pretrain weight for 2D detector.
   - **3D Joint detectors**
      - Please Download PoseTriplet's pretrained model **ckpt_ep_045.bin** from [here](https://drive.google.com/file/d/1oonIlBBXT44maCGYR6XJdGyGmtlG3jBQ/view?usp=sharing),
        and place it into `./checkpoint` folder

## Usage

1. Please place your video into `./wild_eval/source_video` folder. (We've prepared a test video).
2. Run **videopose-j16-wild-eval_run.py**. Then you will find the rendered output video in `./wild_eval` folder.

## Acknowledgement

The inference code is adapted from [video-to-pose3d](https://github.com/zh-plus/video-to-pose3D)
