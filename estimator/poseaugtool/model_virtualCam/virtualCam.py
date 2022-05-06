import torch
import torchgeometry as tgm
from torch import nn
from poseaugtool.model_conv1d.conv1d import Conv1dBlock
from common.quaternion import qrot
from common.camera import project_to_2d_purelinear

class DoubleLinear(nn.Module):
    def __init__(self, linear_size):
        super(DoubleLinear, self).__init__()
        self.w1 = nn.Linear(linear_size, linear_size)
        self.batch_norm1 = nn.BatchNorm1d(linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)
        self.batch_norm2 = nn.BatchNorm1d(linear_size)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)

        return y


class Dis_Conv1D(nn.Module):
    def __init__(self, nx, ks=3, nh_conv1d=[64, 64]):
        super(Dis_Conv1D, self).__init__()
        self.nx = nx
        self.nh_conv1d = nh_conv1d   # hidden dim of conv1d
        # encode
        self.conv1 = Conv1dBlock(nx, nh_conv1d, activation='leak', ks=ks)
        self.out = nn.Conv1d(nh_conv1d[-1], 1, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        :param x: B x T x jd ---> B x jd x T --> B x nh
        :return: B
        '''
        if len(x.shape) == 4:
            'B x T x 16 x 3'
            b, t, j, d = x.shape
            x = x.view(b, t, j*d)

        x = x.permute(0, 2, 1).contiguous()  # B x T x jd ---> B x jd x T
        hs_x = self.conv1(x)
        hs_x = self.out(hs_x)
        hs_x = torch.mean(hs_x, dim=-1)  # B x nh x Ti ---> B x nh
        return hs_x

#######################################################################################
# ####### gan generator for virtual camera
#######################################################################################
class G_camera(nn.Module):
    """
    v0
    """
    def __init__(self, args, nx=48, ks=3, noise_channle=64):
        super(G_camera, self).__init__()
        self.cam_r_range = args.cam_r_range
        self.cam_t_range = args.cam_t_range
        self.noise_channle = noise_channle

        nh_conv1d = [64, 64]   # hidden dim of conv1d
        self.conv1 = Conv1dBlock(nx, nh_conv1d, activation='leak', ks=ks)

        linear_size = noise_channle + nh_conv1d[-1]
        self.wr = nn.Sequential(
            DoubleLinear(linear_size),
            nn.Linear(linear_size, 3),
            nn.Tanh()
        )
        self.wt = nn.Sequential(
            DoubleLinear(linear_size),
            nn.Linear(linear_size, 3),
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pose3D, noise_dict=None):
        '''
        :param pose3D: B x T x j x d, with non-zero root position
        :return: dict B x T x j x d
        '''
        x = pose3D * 1.
        if len(x.shape) == 4:
            'B x T x 16 x 3'
            b, t, j, d = x.shape
            x = x.view(b, t, j*d)
        # get the feature for RT
        x = x.permute(0, 2, 1).contiguous()  # B x T x jd ---> B x jd x T
        x = self.conv1(x)  # B x c x T
        x = torch.mean(x, dim=-1)  # B x nh x Ti ---> B x nh

        # caculate R - QR
        # noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        if noise_dict is None:
            noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        else:
            noise = noise_dict['G-cam-r']

        r = self.wr(torch.cat((x, noise), dim=1)) * self.cam_r_range
        r = r.view(r.size(0), 3)
        mask = torch.ones_like(r)
        mask[:, 1:] = 0
        r = r * mask

        qr = tgm.angle_axis_to_quaternion(r)

        # caculate T
        # noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        if noise_dict is None:
            noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        else:
            noise = noise_dict['G-cam-t']

        tmp_t = self.wt(torch.cat((x, noise), dim=1))
        tx = self.tanh(tmp_t[:, :1]) * self.cam_t_range * 1.
        ty = self.tanh(tmp_t[:, 1:2]) * self.cam_t_range * 0.5
        tz = self.sigmoid(tmp_t[:, 2:]) * self.cam_t_range * 1. + 2.
        t = torch.cat([tx, ty, tz], dim=1)

        # use R T create new 2D-3D pair
        pose3D_camed = qrot(qr.unsqueeze(1).unsqueeze(1).repeat(1, *pose3D.shape[1:-1], 1), pose3D) \
                       + t.unsqueeze(1).unsqueeze(1).repeat(1, *pose3D.shape[1:-1], 1)
        pose2D_camed = project_to_2d_purelinear(pose3D_camed)
        return {
            'pose3D_camed': pose3D_camed,
            'pose2D_camed': pose2D_camed,
            'r': r,
            't': t,
        }


class G_camera_v2(nn.Module):
    """
    v2
    """
    def __init__(self, args, nx=48, ks=3, noise_channle=64):
        super(G_camera_v2, self).__init__()
        self.cam_r_range = args.cam_r_range
        self.cam_t_range = args.cam_t_range
        self.noise_channle = noise_channle

        nh_conv1d = [64, 64]   # hidden dim of conv1d
        self.conv1 = Conv1dBlock(nx, nh_conv1d, activation='leak', ks=ks)

        linear_size = noise_channle + nh_conv1d[-1]
        self.wr = nn.Sequential(
            DoubleLinear(linear_size),
            nn.Linear(linear_size, 3),
            nn.Tanh()
        )
        self.wt = nn.Sequential(
            DoubleLinear(linear_size),
            nn.Linear(linear_size, 3),
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pose3D, noise_dict=None):
        '''
        :param pose3D: B x T x j x d, with non-zero root position
        :return: dict B x T x j x d
        '''
        x = pose3D * 1.
        if len(x.shape) == 4:
            'B x T x 16 x 3'
            b, t, j, d = x.shape
            x = x.view(b, t, j*d)
        # get the feature for RT
        x = x.permute(0, 2, 1).contiguous()  # B x T x jd ---> B x jd x T
        x = self.conv1(x)  # B x c x T
        x = torch.mean(x, dim=-1)  # B x nh x Ti ---> B x nh

        # caculate R - QR
        # noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        if noise_dict is None:
            noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        else:
            noise = noise_dict['G-cam-r']

        r = self.wr(torch.cat((x, noise), dim=1)) * self.cam_r_range
        r = r.view(r.size(0), 3)
        # mask = torch.ones_like(r)
        # mask[:, 1:] = 0
        # r = r * mask

        qr = tgm.angle_axis_to_quaternion(r)

        # caculate T
        # noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        if noise_dict is None:
            noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        else:
            noise = noise_dict['G-cam-t']

        tmp_t = self.wt(torch.cat((x, noise), dim=1))
        tx = self.tanh(tmp_t[:, :1]) * self.cam_t_range * 1.
        ty = self.tanh(tmp_t[:, 1:2]) * self.cam_t_range * 1. # 0922 0.5
        tz = self.sigmoid(tmp_t[:, 2:]) * self.cam_t_range * 1. + 2.
        t = torch.cat([tx, ty, tz], dim=1)

        # use R T create new 2D-3D pair
        pose3D_camed = qrot(qr.unsqueeze(1).unsqueeze(1).repeat(1, *pose3D.shape[1:-1], 1), pose3D) \
                       + t.unsqueeze(1).unsqueeze(1).repeat(1, *pose3D.shape[1:-1], 1)
        pose2D_camed = project_to_2d_purelinear(pose3D_camed)
        return {
            'pose3D_camed': pose3D_camed,
            'pose2D_camed': pose2D_camed,
            'r': r,
            't': t,
        }





################################################################################################
############################# dis 2D      #####################################
################################################################################################
from function.gan_utils import diff


class Pose2DVideoDiscriminator(nn.Module):
    def __init__(self, ks=3, nh_conv1d=[64, 64]):
        super(Pose2DVideoDiscriminator, self).__init__()
        # only check on bone angle, not bone vector.
        num_joints = 16
        self.num_joints = num_joints
        self.traj_path = Dis_Conv1D(16*3, ks, nh_conv1d=nh_conv1d)

    def forward(self, inputs_2d):
        '''
        inputs_2d: B x T x 16 x 2
        '''
        if len(inputs_2d.shape) == 3 and inputs_2d.shape[-1] == self.num_joints * 2:
            'B x T x 48'
            b, t, jd = inputs_2d.shape
            inputs_2d = inputs_2d.view(b, t, self.num_joints, 2)

        b, t, j, d = inputs_2d.shape
        assert j == self.num_joints

        #################
        traj_velocity = diff(inputs_2d)
        traj_velocity = torch.norm(traj_velocity, dim=3, keepdim=True)

        traj_x = torch.cat([inputs_2d.reshape(b, t, -1),
                                traj_velocity.reshape(b, t, -1),
                                ], dim=2)

        out = self.traj_path(traj_x)
        return out

class Pose2DVideoDiscriminatorV2(nn.Module):
    def __init__(self, ks=3, nh_conv1d=[64, 64]):
        super(Pose2DVideoDiscriminatorV2, self).__init__()
        # only check on bone angle, not bone vector.
        num_joints = 16
        self.num_joints = num_joints
        self.traj_path = Dis_Conv1D(16*2, ks, nh_conv1d=nh_conv1d)

    def forward(self, inputs_2d):
        '''
        inputs_2d: B x T x 16 x 2
        '''
        if len(inputs_2d.shape) == 3 and inputs_2d.shape[-1] == self.num_joints * 2:
            'B x T x 48'
            b, t, jd = inputs_2d.shape
            inputs_2d = inputs_2d.view(b, t, self.num_joints, 2)

        b, t, j, d = inputs_2d.shape
        assert j == self.num_joints
        #################
        out = self.traj_path(inputs_2d)
        return out

# MLP version
class Pos2dPairDiscriminator(nn.Module):
    def __init__(self, num_joints=16, d_ch_num=64):  # d_ch_num=100 default
        super(Pos2dPairDiscriminator, self).__init__()

        # Pose path
        self.pose_layer_1 = nn.Linear(num_joints*2*2, d_ch_num)
        self.pose_layer_2 = nn.Linear(d_ch_num, d_ch_num)
        self.pose_layer_3 = nn.Linear(d_ch_num, d_ch_num)
        # self.pose_layer_4 = nn.Linear(d_ch_num, d_ch_num)

        self.layer_last = nn.Linear(d_ch_num, d_ch_num)
        self.layer_pred = nn.Linear(d_ch_num, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x_in):
        """
        input: b x 2 x 16 x 2
        """
        # Pose path
        x = x_in[:, [0, -1]] * 1.  # only use the end frame
        # x[:, :, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]] = x.clone()[:, :, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]] * 0.
        x = x.contiguous().view(x.size(0), -1)
        d = self.relu(self.pose_layer_1(x))
        d = self.relu(self.pose_layer_2(d))
        # d = self.relu(self.pose_layer_3(d) + d)
        # d = self.pose_layer_4(d)

        d_last = self.relu(self.layer_last(d))
        d_out = self.layer_pred(d_last)

        return d_out


class Pos2dPairDiscriminator_v5(nn.Module):
    def __init__(self, num_joints=16, d_ch_num=64):  # d_ch_num=100 default
        super(Pos2dPairDiscriminator_v5, self).__init__()

        # Pose path
        self.pose_layer_1 = nn.Linear(num_joints*2*2, d_ch_num)
        self.bn_layer_1 = nn.BatchNorm1d(d_ch_num)
        self.pose_layer_2 = nn.Linear(d_ch_num, d_ch_num)
        self.bn_layer_2 = nn.BatchNorm1d(d_ch_num)
        # self.pose_layer_3 = nn.Linear(d_ch_num, d_ch_num)
        # self.bn_layer_3 = nn.BatchNorm1d(d_ch_num)
        # self.pose_layer_4 = nn.Linear(d_ch_num, d_ch_num)

        self.layer_last = nn.Linear(d_ch_num, d_ch_num)
        self.bn_last = nn.BatchNorm1d(d_ch_num)
        self.layer_pred = nn.Linear(d_ch_num, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x_in):
        """
        input: b x 2 x 16 x 2
        """
        # Pose path
        x = x_in[:, [0, -1]] * 1.  # only use the end frame
        # x[:, :, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]] = x.clone()[:, :, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]] * 0.
        x = x.contiguous().view(x.size(0), -1)
        d = self.relu(self.bn_layer_1(self.pose_layer_1(x)))
        d = self.relu(self.bn_layer_2(self.pose_layer_2(d)))
        # d = self.relu(self.pose_layer_3(d) + d)
        # d = self.pose_layer_4(d)

        d_last = self.relu(self.bn_last(self.layer_last(d)))
        d_out = self.layer_pred(d_last)

        return d_out


from function.gan_utils import get_BoneVecbypose3d
class Pos2dPairDiscriminator_v6(nn.Module):
    def __init__(self, num_joints=16, d_ch_num=16):  # d_ch_num=100 default
        super(Pos2dPairDiscriminator_v6, self).__init__()

        self.joint_idx_toD = [0]
        num_joints = len(self.joint_idx_toD)
        self.bv_idx_toD = [6]
        num_jbv = len(self.bv_idx_toD)

        # Pose path
        self.pose_layer_1 = nn.Linear((num_joints+num_jbv)*2*2, d_ch_num)
        self.bn_layer_1 = nn.BatchNorm1d(d_ch_num)
        self.pose_layer_2 = nn.Linear(d_ch_num, d_ch_num)
        self.bn_layer_2 = nn.BatchNorm1d(d_ch_num)
        # self.pose_layer_3 = nn.Linear(d_ch_num, d_ch_num)
        # self.bn_layer_3 = nn.BatchNorm1d(d_ch_num)
        # self.pose_layer_4 = nn.Linear(d_ch_num, d_ch_num)

        self.layer_last = nn.Linear(d_ch_num, d_ch_num)
        self.bn_last = nn.BatchNorm1d(d_ch_num)
        self.layer_pred = nn.Linear(d_ch_num, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x_in):
        """
        input: b x 2 x 16 x 2
        """
        # Pose path
        sz = x_in.shape
        x_bv = get_BoneVecbypose3d(x_in.reshape(-1, 16, 2)).reshape(sz[0], sz[1], 15, 2)
        x1 = x_bv[:, [0, -1], self.bv_idx_toD] * 1.  # only use the end frame
        x2 = x_in[:, [0, -1], self.joint_idx_toD] * 1.  # only use the end frame
        x = torch.cat([x1, x2], dim=2)
        # x[:, :, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]] = x.clone()[:, :, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]] * 0.
        x = x.contiguous().view(x.size(0), -1)
        d = self.relu(self.bn_layer_1(self.pose_layer_1(x)))
        d = self.relu(self.bn_layer_2(self.pose_layer_2(d)))
        # d = self.relu(self.pose_layer_3(d) + d)
        # d = self.pose_layer_4(d)

        d_last = self.relu(self.bn_last(self.layer_last(d)))
        d_out = self.layer_pred(d_last)

        return d_out



if __name__ == '__main__':
    d = Dis_Conv1D(48, 3)
    input = torch.zeros(64, 32, 48)  # B x T x J3
    out = d(input)
    print('out: ', out.shape)

    d = Pose2DVideoDiscriminator(3)
    input = torch.zeros(64, 75, 16, 2)  # B x T x J3
    out = d(input)
    print('out: ', out.shape)

