import os

import numpy as np

from bvh_skeleton import humanoid_1205_skeleton


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

    Converter = humanoid_1205_skeleton.SkeletonConverter()
    prediction3dpoint = Converter.convert_to_21joint(prediction3dpoint)

    human36m_skeleton = humanoid_1205_skeleton.H36mSkeleton()
    human36m_skeleton.poses2bvh(prediction3dpoint, output_file=bvhfileName)


if __name__ == '__main__':
    """
    test pose2bvh
    """

    # prediction = np.load('bvh_skeleton/test_data/0626_take_01_sktwpos_16joints.npy', allow_pickle=True)
    # prediction = np.load('bvh_skeleton/test_data/h36m_take_599_predicted_3d_wpos.npy', allow_pickle=True)[690:710]
    prediction = np.load('../../PoseTriplet-test/estimator_inference/wild_eval/pred3D_pose/bilibili-clip/kunkun_clip_pred3D.pkl', allow_pickle=True)['result']
    prediction = prediction.astype('float32')

    # convert pose to bvh
    # viz_output = 'outputs/outputvideo/alpha_result.mp4'
    # viz_output = 'bvh_skeleton/test_data/0626_take_01.mp4'
    viz_output = 'bvh_skeleton/test_data/kunkun_clip.mp4'
    prediction_copy = np.copy(prediction)
    write_standard_bvh(viz_output, prediction_copy)
    # use bvhacker to view the result
