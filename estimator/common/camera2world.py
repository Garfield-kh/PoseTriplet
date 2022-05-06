import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

"""
camera to world:
case 1: assume the camera ID is known for each clip.
A. do PCA for 150 clips to found a gravity approximation
B. assume a stand pose, filter out the stand pose from prediction, do PCA among those.
case 2: assume a rough gravity direction
A. do PCA for each clip, choose the gravity approximation by compare them.
case 3: for in the wild scenario
A. manual set
B. do case 1 -> B
case 4: use off-the-shelf ground/gravity estimation
TBD
"""

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector, axis=-1, keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1/M1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


def wxyz2xyzw(wfist):
    "convert w x y z to x y z w, xyzw is used in scipy"
    return np.stack([wfist[1], wfist[2], wfist[3], wfist[0]], axis=0)

def xyzw2wxyz(wfist):
    "convert x y z w to w x y z"
    return np.stack([wfist[3], wfist[0], wfist[1], wfist[2]], axis=0)


def get_pca_components(pose_cam_in):
    """
    input: tx16x3 pose seq in camera coordinate
    return: ZXY axis
    """
    #     x = expert_dict[takes[30]]['predicted_3d']
    #     print(x.shape)
    x = pose_cam_in - pose_cam_in[:, :1, :]
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x.reshape(-1, 3)).reshape(x.shape)
    pca_zxy = pca.components_
    pca_xyz = pca_zxy[[1, 2, 0], :]
    return unit_vector(pca_xyz)


def check_Z_dir(pose_cam_in, pca_xyz):
    """Check the Z direction is up or down"""
    pca_x = pca_xyz[0] * 1.
    pca_y = pca_xyz[1] * 1.
    pca_z = pca_xyz[2] * 1.

    fix_z = [0, -0.97, -0.25]  # the appropriate direction
    fix_z = unit_vector(fix_z)
    dot_avg = angle_between(fix_z, pca_z)

    if dot_avg > 0.996:
        pca_z = pca_z * +1
        pca_x = np.cross(pca_y, pca_z)
    elif dot_avg < -0.996:
        pca_z = pca_z * -1
        pca_x = np.cross(pca_y, pca_z)
    else:
        pca_z = fix_z * +1
        #         pca_x = [1, 0, 0]
        pca_y = np.cross(pca_z, pca_x)
        pca_x = np.cross(pca_y, pca_z)
    new_pca_xyz = np.stack([pca_x, pca_y, pca_z])
    return new_pca_xyz

# def check_Z_accuracy(pca_xyz, cam_ex):
#     """
#     a double check for z direction
#     """
#     pca_z = pca_xyz[2] * 1.
#     #     q = expert_dict[takes[30]]['cam_ex']
#     q = cam_ex * 1.
#     r_cam2world = R.from_quat(wxyz2xyzw(q))  # .inv()
#     world_z = r_cam2world.inv().apply([0, 0, 1])
#
#     acc = angle_between(pca_z, world_z)
#     if np.abs(acc) > 0.98:
#         pass
#     else:
#         assert False, "the pca_z seems wrong with value {}, please check!!!".format(acc)


def cam2world_byPCA(pose_cam_in, cam_ex=None):
    # input the pose in camera space, then do pca, get the zxy axis,
    pca_xyz_incam = get_pca_components(pose_cam_in)
    # check the leg direction, assume leg is always down,
    pca_xyz_incam = check_Z_dir(pose_cam_in, pca_xyz_incam)
    # calculate the Z direction in camera space
    # check_Z_accuracy(pca_xyz_incam, cam_ex)
    # get a rotation matrix by the world Z direction
    world2cam_bypca = R.from_matrix(pca_xyz_incam.T)
    cam2world_bypca = world2cam_bypca.inv()
    # rotate the pose from camera space to world space
    pose_world_out = cam2world_bypca.apply(pose_cam_in.reshape(-1, 3)).reshape(pose_cam_in.shape)
    return pose_world_out

def camera_to_worldByPCA(X, R=None):
    pose_cam_in = X.detach().cpu().numpy()[0]
    cam_ex = R.detach().cpu().numpy()[0] if R is not None else R
    pose_world_out = cam2world_byPCA(pose_cam_in, cam_ex)
    return torch.from_numpy(pose_world_out.astype('float32')).unsqueeze(0)

