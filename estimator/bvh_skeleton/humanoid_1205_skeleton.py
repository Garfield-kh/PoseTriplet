from . import math3d
# from . import math3dkh as math3d  # scipy too slow
from . import bvh_helper
# from . import math3dV1  # for debug

import numpy as np
from scipy.spatial.transform import Rotation

"""
step1: convert 16 joints to 21 joints
step2: save 21 joints to bvh
"""


class SkeletonConverter(object):

    def __init__(self):
        self.root = 'Hips'
        self.keypoint2index = {
            'Hips': 0,
            'RightUpLeg': 1,
            'RightLeg': 2,
            'RightFoot': 3,
            'LeftUpLeg': 4,
            'LeftLeg': 5,
            'LeftFoot': 6,
            'Spine2': 7,
            # 'Spine3': 8,
            'Neck': 8,
            'Head': 9,
            'LeftArm': 10,
            'LeftForeArm': 11,
            'LeftHand': 12,
            'RightArm': 13,
            'RightForeArm': 14,
            'RightHand': 15,
            # 'RightFootEndSite': -1,
            # 'LeftFootEndSite': -1,
            # 'LeftHandEndSite': -1,
            # 'RightHandEndSite': -1
        }

        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        # new joint index keypoint setting
        self.keypoint2index_21joint = {
            'Hips': 0,
            'RightUpLeg': 1,
            'RightLeg': 2,
            'RightFoot': 3,
            'LeftUpLeg': 4,
            'LeftLeg': 5,
            'LeftFoot': 6,
            'Spine': 7,
            'Spine1': 8,
            'Spine2': 9,
            'Spine3': 10,
            'Neck': 11,
            'Head': 12,
            'LeftShoulder': 13,
            'LeftArm': 14,
            'LeftForeArm': 15,
            'LeftHand': 16,
            'RightShoulder': 17,
            'RightArm': 18,
            'RightForeArm': 19,
            'RightHand': 20,
            # 'RightFootEndSite': -1,
            # 'LeftFootEndSite': -1,
            # 'LeftHandEndSite': -1,
            # 'RightHandEndSite': -1
        }

        self.index2keypoint_21joint = {v: k for k, v in self.keypoint2index_21joint.items()}
        self.keypoint_num_21joint = len(self.keypoint2index_21joint)

    def convert_to_21joint(self, poses_3d):
        """
        add spine, spine1, spine3, head end site, LeftShoulder, RightShoulder
        poses_3d: tx16x3
        :return:
        """
        tmp_poses_dict = {}
        """ spine, spine1 <- Hips,Spine2 """
        vec_Hips2Spine2 = poses_3d[:, self.keypoint2index['Spine2']] - poses_3d[:, self.keypoint2index['Hips']]
        tmp_poses_dict['Spine'] = poses_3d[:, self.keypoint2index['Hips']] + 1 / 3 * vec_Hips2Spine2
        tmp_poses_dict['Spine1'] = poses_3d[:, self.keypoint2index['Hips']] + 2 / 3 * vec_Hips2Spine2
        """ spine3 <- Spine2, Neck"""
        vec_Spine22Neck = poses_3d[:, self.keypoint2index['Neck']] - poses_3d[:, self.keypoint2index['Spine2']]
        tmp_poses_dict['Spine3'] = poses_3d[:, self.keypoint2index['Spine2']] + 1 / 2 * vec_Spine22Neck
        """ LeftShoulder <- Neck,  LeftArm"""
        vec_Neck2LeftArm = poses_3d[:, self.keypoint2index['LeftArm']] - poses_3d[:, self.keypoint2index['Neck']]
        tmp_poses_dict['LeftShoulder'] = poses_3d[:, self.keypoint2index['Neck']] + 1 / 6 * vec_Neck2LeftArm
        """ RightShoulder <- Neck,  RightArm"""
        vec_Neck2RightArm = poses_3d[:, self.keypoint2index['RightArm']] - poses_3d[:, self.keypoint2index['Neck']]
        tmp_poses_dict['RightShoulder'] = poses_3d[:, self.keypoint2index['Neck']] + 1 / 6 * vec_Neck2RightArm

        """ expand current tmp_poses_dict """
        for keypoint in self.keypoint2index:
            tmp_poses_dict[keypoint] = poses_3d[:, self.keypoint2index[keypoint]]

        """ re-order the joint"""
        poses_3d_21joint = np.zeros((poses_3d.shape[0], 21, 3), dtype='float32')
        for idx in self.index2keypoint_21joint:
            poses_3d_21joint[:, idx] = tmp_poses_dict[self.index2keypoint_21joint[idx]]
        return poses_3d_21joint


class H36mSkeleton(object):

    def __init__(self):
        self.root = 'Hips'

        self.keypoint2index = {
            'Hips': 0,
            'RightUpLeg': 1,
            'RightLeg': 2,
            'RightFoot': 3,
            'LeftUpLeg': 4,
            'LeftLeg': 5,
            'LeftFoot': 6,
            'Spine': 7,
            'Spine1': 8,
            'Spine2': 9,
            'Spine3': 10,
            'Neck': 11,
            'Head': 12,
            'HeadEndSite': -1,
            'LeftShoulder': 13,
            'LeftArm': 14,
            'LeftForeArm': 15,
            'LeftHand': 16,
            'RightShoulder': 17,
            'RightArm': 18,
            'RightForeArm': 19,
            'RightHand': 20,
            'RightFootEndSite': -1,
            'LeftFootEndSite': -1,
            'LeftHandEndSite': -1,
            'RightHandEndSite': -1
        }

        self.index2keypoint = {v: k for k, v in self.keypoint2index.items()}
        self.keypoint_num = len(self.keypoint2index)

        self.children = {
            'Hips': ['RightUpLeg', 'LeftUpLeg', 'Spine'],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightFootEndSite'],
            'RightFootEndSite': [],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftFootEndSite'],
            'LeftFootEndSite': [],
            'Spine': ['Spine1'],
            'Spine1': ['Spine2'],
            'Spine2': ['Spine3'],
            'Spine3': ['Neck', 'LeftShoulder', 'RightShoulder'],
            'Neck': ['Head'],
            'Head': ['HeadEndSite'],
            'HeadEndSite': [],
            'LeftShoulder': ['LeftArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand'],
            'LeftHand': ['LeftHandEndSite'],
            'LeftHandEndSite': [],
            'RightShoulder': ['RightArm'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'RightHand': ['RightHandEndSite'],
            'RightHandEndSite': []
        }

        self.parent = {self.root: None}
        for parent, children in self.children.items():
            for child in children:
                self.parent[child] = parent

        self.left_joints = [
            joint for joint in self.keypoint2index
            if 'Left' in joint
        ]
        self.right_joints = [
            joint for joint in self.keypoint2index
            if 'Right' in joint
        ]

        # define the offset pose
        self.initial_directions = {
            'Hips': [0, 0, 0],
            'RightUpLeg': [-1, 0, 0],
            'RightLeg': [0, 0, -1],
            'RightFoot': [0, 0, -1],
            'RightFootEndSite': [0, -1, 0],
            'LeftUpLeg': [1, 0, 0],
            'LeftLeg': [0, 0, -1],
            'LeftFoot': [0, 0, -1],
            'LeftFootEndSite': [0, -1, 0],
            'Spine': [0, 0, 1],
            'Spine1': [0, 0, 1],
            'Spine2': [0, 0, 1],
            'Spine3': [0, 0, 1],
            'Neck': [0, 0, 1],
            'Head': [0, 0, 1],
            'HeadEndSite': [0, 0, 1],
            'LeftShoulder': [1, 0, 0],
            'LeftArm': [1, 0, 0],
            'LeftForeArm': [1, 0, 0],
            'LeftHand': [1, 0, 0],
            'LeftHandEndSite': [1, 0, 0],
            'RightShoulder': [-1, 0, 0],
            'RightArm': [-1, 0, 0],
            'RightForeArm': [-1, 0, 0],
            'RightHand': [-1, 0, 0],
            'RightHandEndSite': [-1, 0, 0]
        }

        self.dcms = {}
        for joint in self.keypoint2index:
            self.dcms[joint] = None

    def get_initial_offset(self, poses_3d):
        # TODO: RANSAC
        bone_lens = {self.root: [0]}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            p_idx = self.keypoint2index[parent]
            for child in self.children[parent]:
                if 'EndSite' in child:
                    bone_lens[child] = 0.4 * bone_lens[parent]
                    continue
                stack.append(child)

                c_idx = self.keypoint2index[child]
                bone_lens[child] = np.linalg.norm(
                    poses_3d[:, p_idx] - poses_3d[:, c_idx],
                    axis=1
                )

        bone_len = {}
        for joint in self.keypoint2index:
            if 'Left' in joint or 'Right' in joint:
                base_name = joint.replace('Left', '').replace('Right', '')
                left_len = np.mean(bone_lens['Left' + base_name])
                right_len = np.mean(bone_lens['Right' + base_name])
                bone_len[joint] = (left_len + right_len) / 2
            else:
                bone_len[joint] = np.mean(bone_lens[joint])

        initial_offset = {}
        for joint, direction in self.initial_directions.items():
            direction = np.array(direction) / max(np.linalg.norm(direction), 1e-12)
            initial_offset[joint] = direction * bone_len[joint]

        return initial_offset

    def get_bvh_header(self, poses_3d):
        initial_offset = self.get_initial_offset(poses_3d)

        nodes = {}
        for joint in self.keypoint2index:
            is_root = joint == self.root
            is_end_site = 'EndSite' in joint
            nodes[joint] = bvh_helper.BvhNode(
                name=joint,
                offset=initial_offset[joint],
                rotation_order='xyz' if not is_end_site else '',  # default zxy
                is_root=is_root,
                is_end_site=is_end_site,
            )
        for joint, children in self.children.items():
            nodes[joint].children = [nodes[child] for child in children]
            for child in children:
                nodes[child].parent = nodes[joint]

        header = bvh_helper.BvhHeader(root=nodes[self.root], nodes=nodes)
        return header

    def pose2euler(self, pose, header):
        channel = []
        quats = {}
        # quatsV1 = {}
        eulers = {}
        stack = [header.root]

        # check is hand in singularity.
        index = self.keypoint2index
        LeftForeArm_angle = math3d.anglefrom3points(pose[index['LeftArm']], pose[index['LeftForeArm']],
                                                    pose[index['LeftHand']])
        LeftForeArm_straight = np.abs(LeftForeArm_angle - 180) < 10
        RightForeArm_angle = math3d.anglefrom3points(pose[index['RightArm']], pose[index['RightForeArm']],
                                                     pose[index['RightHand']])
        RightForeArm_straight = np.abs(RightForeArm_angle - 180) < 10

        while stack:
            node = stack.pop()
            joint = node.name
            joint_idx = self.keypoint2index[joint]

            if node.is_root:
                channel.extend(pose[joint_idx])

            index = self.keypoint2index
            order = None
            if joint == 'Hips':
                x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
                y_dir = None
                z_dir = pose[index['Spine']] - pose[joint_idx]
                order = 'zyx'
            elif joint in ['RightUpLeg', 'RightLeg']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['Hips']] - pose[index['RightUpLeg']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'

            elif joint in ['LeftUpLeg', 'LeftLeg']:
                child_idx = self.keypoint2index[node.children[0].name]
                x_dir = pose[index['LeftUpLeg']] - pose[index['Hips']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[child_idx]
                order = 'zyx'


            elif joint == 'Spine':
                x_dir = pose[index['LeftUpLeg']] - pose[index['RightUpLeg']]
                y_dir = None
                z_dir = pose[index['Spine1']] - pose[joint_idx]
                order = 'zyx'

            elif joint == 'Spine2':
                x_dir = pose[index['LeftArm']] - \
                    pose[index['RightArm']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[index['Spine1']]
                order = 'zyx'
            elif joint == 'Spine3':
                x_dir = pose[index['LeftArm']] - \
                        pose[index['RightArm']]
                y_dir = None
                z_dir = pose[joint_idx] - pose[index['Spine2']]
                order = 'zyx'

            elif joint == 'Neck':
                x_dir = None
                y_dir = pose[index['Spine3']] - pose[joint_idx]
                z_dir = pose[index['Head']] - pose[index['Spine3']]
                order = 'zxy'
            elif joint == 'LeftShoulder':
                x_dir = pose[index['LeftArm']] - pose[joint_idx]
                y_dir = pose[index['LeftArm']] - pose[index['LeftForeArm']]
                z_dir = None
                order = 'xzy'
            elif joint == 'LeftArm':
                if LeftForeArm_straight and self.dcms['LeftForeArm'] is not None:
                    """in case of singularity, use z from forearm"""
                    x_dir = pose[index['LeftForeArm']] - pose[joint_idx]
                    y_dir = None
                    z_dir = self.dcms['LeftForeArm'][2] * 1.
                    order = 'xyz'
                else:
                    x_dir = pose[index['LeftForeArm']] - pose[joint_idx]
                    y_dir = pose[index['LeftForeArm']] - pose[index['LeftHand']]
                    z_dir = None
                    order = 'xzy'
            elif joint == 'LeftForeArm':
                if LeftForeArm_straight and self.dcms['LeftForeArm'] is not None:
                    """in case of singularity, use z from forearm"""
                    x_dir = pose[index['LeftHand']] - pose[joint_idx]
                    y_dir = None
                    z_dir = self.dcms['LeftForeArm'][2] * 1.
                    order = 'xyz'
                else:
                    x_dir = pose[index['LeftHand']] - pose[joint_idx]
                    y_dir = pose[joint_idx] - pose[index['LeftArm']]
                    z_dir = None
                    order = 'xzy'
            elif joint == 'RightShoulder':
                x_dir = pose[joint_idx] - pose[index['RightArm']]
                y_dir = pose[index['RightArm']] - pose[index['RightForeArm']]
                z_dir = None
                order = 'xzy'
            elif joint == 'RightArm':
                if RightForeArm_straight and self.dcms['RightForeArm'] is not None:
                    """in case of singularity, use z from forearm"""
                    x_dir = pose[joint_idx] - pose[index['RightForeArm']]
                    y_dir = None
                    z_dir = self.dcms['RightForeArm'][2] * 1.
                    order = 'xyz'
                else:
                    x_dir = pose[joint_idx] - pose[index['RightForeArm']]
                    y_dir = pose[index['RightForeArm']] - pose[index['RightHand']]
                    z_dir = None
                    order = 'xzy'
            elif joint == 'RightForeArm':
                if RightForeArm_straight and self.dcms['RightForeArm'] is not None:
                    """in case of singularity, use z from forearm"""
                    x_dir = pose[joint_idx] - pose[index['RightHand']]
                    y_dir = None
                    z_dir = self.dcms['RightForeArm'][2] * 1.
                    order = 'xyz'
                else:
                    x_dir = pose[joint_idx] - pose[index['RightHand']]
                    y_dir = pose[joint_idx] - pose[index['RightArm']]
                    z_dir = None
                    order = 'xzy'
            if order:
                dcm = math3d.dcm_from_axis(x_dir, y_dir, z_dir, order)  # 3x3 [axis['x'], axis['y'], axis['z']]
                self.dcms[joint] = dcm.copy()
                quats[joint] = math3d.dcm2quat(dcm)
            else:
                quats[joint] = quats[self.parent[joint]].copy()

            local_quat = quats[joint].copy()
            if node.parent:
                local_quat = math3d.quat_divide(
                    q=quats[joint], r=quats[node.parent.name]
                )

            euler = math3d.quat2euler(
                q=local_quat, order=node.rotation_order
            )

            # reset shoulder x rotation to 0, and update its local_quat ---> quats.
            if joint in ['LeftShoulder', 'RightShoulder', 'Neck']:
                tmp_idx = 2 if joint == 'Neck' else 0
                euler[tmp_idx] = tmp_idx  # 3
                local_quat = math3d.euler2quat(euler)
                # use local_quat * parents_quat -> quat in world coord.
                quat = math3d.quat_mul(quats[node.parent.name], local_quat)
                # update quats[joint]
                quats[joint] = quat * 1.

            euler = np.rad2deg(euler)

            eulers[joint] = euler
            channel.extend(euler)

            for child in node.children[::-1]:
                if not child.is_end_site:
                    stack.append(child)

        return channel

    def poses2bvh(self, poses_3d, header=None, output_file=None):
        if not header:
            header = self.get_bvh_header(poses_3d)

        channels = []
        for frame, pose in enumerate(poses_3d):
            channels.append(self.pose2euler(pose, header))

        if output_file:
            bvh_helper.write_bvh(output_file, header, channels)

        return channels, header



import os


# convert 3D pose to bvh format and save at outputs/outputvideo/alpha_pose_xxx/bvh
def write_standard_bvh(outbvhfilepath, prediction3dpoint):
    '''
    :param outbvhfilepath:
    :param prediction3dpoint:
    :return:
    '''

    # scale 100 for bvhacker viewer
    for frame in prediction3dpoint:
        for point3d in frame:
            point3d[0] *= 100
            point3d[1] *= 100
            point3d[2] *= 100

            # exchange y and z for coordinate rotation.
            # X = point3d[0]
            # Y = point3d[1]
            # Z = point3d[2]

            # point3d[0] = -X
            # point3d[1] = Z
            # point3d[2] = Y

    dir_name = os.path.dirname(outbvhfilepath)
    basename = os.path.basename(outbvhfilepath)
    video_name = basename[:basename.rfind('.')]
    bvhfileDirectory = os.path.join(dir_name, video_name, "bvh")
    if not os.path.exists(bvhfileDirectory):
        os.makedirs(bvhfileDirectory)
    bvhfileName = os.path.join(dir_name, video_name, "bvh", "{}_h36m.bvh".format(video_name))

    Converter = SkeletonConverter()
    prediction3dpoint = Converter.convert_to_21joint(prediction3dpoint)

    human36m_skeleton = H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)


if __name__ == '__main__':
    pass
