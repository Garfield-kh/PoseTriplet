# Estimator

> Train pose estimation model and trajectory estimation model \
> Predict 3d human pose with root trajectory from the available 2D pose data

# Data preparation
## [Human3.6M](http://vision.imar.ro/human3.6m/)
The code for Human3.6M data preparation is the same as [PoseAug/DATASETS.md](https://github.com/jfzhang95/PoseAug/blob/main/DATASETS.md). \
Once prepared, the folder will be like:
```
${PoseAug}
├── data
  ├── data_3d_h36m.npz
  ├── data_2d_h36m_gt.npz
  ├── data_2d_h36m_hr.npz
```
This is enough for SSL learning (2D pose) and evaluation (3D pose) in H36M dataset.


## Acknowledgement

The code includes three parts:
1) Estimation models are adapted from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
2) Augmented projection is developed from [PoseAug](https://github.com/jfzhang95/PoseAug).
3) IK part is developed from [video2bvh](https://github.com/KevinLTT/video2bvh).

If you find this part useful for your research, please consider citing them.