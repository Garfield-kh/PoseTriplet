import numpy as np

def world2cam_sktpos(skt_in):
    # t1 = skt_in.reshape(-1, 17, 3)
    # gt_3d_world = np.delete(t1, (8), axis=1)  # remove nose
    gt_3d_world = skt_in.reshape(-1, 16, 3)
    # from x y z to x z -y
    gt_3d = gt_3d_world[..., [0, 2, 1]]
    # gt_3d[:, :, 1] = gt_3d[:, :, 1] * -1
    gt_3d = gt_3d * -1
    return gt_3d

def cam2world_sktpos(skt_in):
    # skint_in: b t j 3
    tmp_skt_in = skt_in * 1.0
    # tmp_skt_in[:, :, :, 1] = tmp_skt_in[:, :, :, 1] * -1
    tmp_skt_in = tmp_skt_in * -1
    gt_3d_world = tmp_skt_in[..., [0, 2, 1]]
    return gt_3d_world