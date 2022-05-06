from __future__ import absolute_import, division, print_function

"""Functions to visualize human poses"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
import os

NTU_BODY_MEMBERS = {
    'left_arm': {'joints': [20, 8, 9, 10, 11, 23], 'side': 'left'},
    'left_fingers': {'joints': [11, 24], 'side': 'left'},
    'right_arm': {'joints': [20, 4, 5, 6, 7, 21], 'side': 'right'},
    'right_fingers': {'joints': [7, 22], 'side': 'right'},
    'head': {'joints': [20, 2, 3], 'side': 'right'},
    'torso': {'joints': [0, 1, 20], 'side': 'right'},
    'left_leg': {'joints': [0, 16, 17, 18, 19], 'side': 'left'},
    'right_leg': {'joints': [0, 12, 13, 14, 15], 'side': 'right'},
}
NTU_NJOINTS = 25
NTU_ACTIONS = ["drink water", "eat meal/snack", "brushing teeth",
               "brushing hair", "drop", "pickup", "throw", "sitting down",
               "standing up (from sitting position)", "clapping", "reading",
               "writing", "tear up paper", "wear jacket", "take off jacket",
               "wear a shoe", "take off a shoe", "wear on glasses",
               "take off glasses", "put on a hat/cap", "take off a hat/cap",
               "cheer up", "hand waving", "kicking something",
               "put something inside pocket / take out something from pocket",
               "hopping (one foot jumping)", "jump up",
               "make a phone call/answer phone", "playing with phone/tablet",
               "typing on a keyboard", "pointing to something with finger",
               "taking a selfie", "check time (from watch)",
               "rub two hands together", "nod head/bow", "shake head",
               "wipe face", "salute", "put the palms together",
               "cross hands in front (say stop)", "sneeze/cough", "staggering",
               "falling", "touch head (headache)",
               "touch chest (stomachache/heart pain)", "touch back (backache)",
               "touch neck (neckache)", "nausea or vomiting condition",
               "use a fan (with hand or paper)/feeling warm",
               "punching/slapping other person", "kicking other person",
               "pushing other person", "pat on back of other person",
               "point finger at the other person", "hugging other person",
               "giving something to other person",
               "touch other person's pocket", "handshaking",
               "walking towards each other", "walking apart from each other"]

MSRC_BODY_MEMBERS = {
    'left_arm': {'joints': [2, 4, 5, 6, 7], 'side': 'left'},
    'right_arm': {'joints': [2, 8, 9, 10, 11], 'side': 'right'},
    'head': {'joints': [1, 2, 3], 'side': 'right'},
    'torso': {'joints': [0, 1], 'side': 'right'},
    'left_leg': {'joints': [0, 12, 13, 14, 15], 'side': 'left'},
    'right_leg': {'joints': [0, 16, 17, 18, 19], 'side': 'right'},
}
MSRC_NJOINTS = 20
MSRC_ACTIONS = ["Start system", "Duck", "Push right",
                "Googles", "Wind it up", "Shoot",
                "Bow", "Throw", "Had enough",
                "Change weapon", "Beat both", "Kick"]

H36_BODY_MEMBERS_FULL = {
    'left_arm': {'joints': [13, 16, 17, 18, 19, 20, 21], 'side': 'left'},
    'left_fingers': {'joints': [19, 22, 23], 'side': 'left'},
    'right_arm': {'joints': [13, 24, 25, 26, 27, 28, 29], 'side': 'right'},
    'right_fingers': {'joints': [27, 30, 31], 'side': 'right'},
    'head': {'joints': [13, 14, 15], 'side': 'right'},
    'torso': {'joints': [0, 11, 12, 13], 'side': 'right'},
    'left_leg': {'joints': [0, 6, 7, 8, 9, 10], 'side': 'left'},
    'right_leg': {'joints': [0, 1, 2, 3, 4, 5], 'side': 'right'},
}
H36_NJOINTS_FULL = 32

H36M_USED_JOINTS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]

H36_BODY_MEMBERS = {
    'left_arm': {'joints': [13, 17, 18, 19], 'side': 'left'},
    'right_arm': {'joints': [13, 25, 26, 27], 'side': 'right'},
    'head': {'joints': [13, 15], 'side': 'right'},
    'torso': {'joints': [0, 12, 13], 'side': 'right'},
    'left_leg': {'joints': [0, 6, 7, 8], 'side': 'left'},
    'right_leg': {'joints': [0, 1, 2, 3], 'side': 'right'},
}

H36_ACTIONS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
               'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
               'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

OPENPOSE_BODY_MEMBERS = {
    'left_arm': {'joints': [2, 3, 4, 3, 2], 'side': 'left'},
    'right_arm': {'joints': [5, 6, 7, 6, 5], 'side': 'right'},
    'head': {'joints': [1, 0, 1], 'side': 'right'},
    # 'ext_head': {'joints': [14, 15, 16, 17, 16, 15, 14], 'side': 'right'},
    'ears': {'joints': [14, 0, 15], 'side': 'right'},
    'torso': {'joints': [2, 1, 5, 1, 8, 1, 11], 'side': 'right'},
    'left_leg': {'joints': [8, 9, 10, 9, 8], 'side': 'left'},
    'right_leg': {'joints': [11, 12, 13, 12, 11], 'side': 'right'},
}
OPENPOSE_NJOINTS = 16


def select_dataset(data_set):
    if data_set == "NTURGBD":
        actions_l = NTU_ACTIONS
        njoints = NTU_NJOINTS
        body_members = NTU_BODY_MEMBERS
    elif data_set == "MSRC12":
        actions_l = MSRC_ACTIONS
        njoints = MSRC_NJOINTS
        body_members = MSRC_BODY_MEMBERS
    elif data_set == "Human36":
        actions_l = H36_ACTIONS
        njoints = len(H36M_USED_JOINTS)
        body_members = H36_BODY_MEMBERS
        new_body_members = {}
        for key, value in body_members.items():
            new_body_members[key] = value.copy()
            new_body_members[key]['joints'] = [H36M_USED_JOINTS.index(j) for j in new_body_members[key]['joints']]
        body_members = new_body_members

    return actions_l, njoints, body_members


class Ax3DPose(object):
    def __init__(self, ax, data_set, lcolor="#3498db", rcolor="#e74c3c"):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        _, self.njoints, self.body_members = select_dataset(data_set)

        self.ax = ax

        # Make connection matrix
        self.plots = {}
        for member in self.body_members.values():
            for j in range(len(member['joints']) - 1):
                j_idx_start = member['joints'][j]
                j_idx_end = member['joints'][j + 1]
                self.plots[(j_idx_start, j_idx_end)] = \
                    self.ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=lcolor if member['side'] == 'left' else rcolor)

        self.plots_mask = []
        for j in range(self.njoints):
            self.plots_mask.append(
                self.ax.plot([0], [0], [0], lw=2, c='black', markersize=8, marker='o', linestyle='dashed',
                             visible=False))

        self.ax.set_xlabel("x", color='r')
        self.ax.set_ylabel("y", color='r')
        self.ax.set_zlabel("z", color='r')

        self.axes_set = False

    def update(self, channels, mask=None, mode='3d'):
        """
        Update the plotted 3d pose.

        Args
          channels: njoints * 3-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert channels.size == self.njoints * 3, \
            "channels should have %d entries, it has %d instead" % (self.njoints * 3, channels.size)
        vals = np.reshape(channels, (self.njoints, -1))

        for member in self.body_members.values():
            for j in range(len(member['joints']) - 1):
                j_idx_start = member['joints'][j]
                j_idx_end = member['joints'][j + 1]
                x = np.array([vals[j_idx_start, 0], vals[j_idx_end, 0]])
                y = np.array([vals[j_idx_start, 1], vals[j_idx_end, 1]])
                z = np.array([vals[j_idx_start, 2], vals[j_idx_end, 2]])
                self.plots[(j_idx_start, j_idx_end)][0].set_xdata(x)
                self.plots[(j_idx_start, j_idx_end)][0].set_ydata(y)
                self.plots[(j_idx_start, j_idx_end)][0].set_3d_properties(z)

        if mask is not None:
            for j in range(self.njoints):
                if mask[j] == 0:
                    self.plots_mask[j][0].set_visible(True)
                else:
                    self.plots_mask[j][0].set_visible(False)
                self.plots_mask[j][0].set_xdata(vals[j, 0])
                self.plots_mask[j][0].set_ydata(vals[j, 1])
                self.plots_mask[j][0].set_3d_properties(vals[j, 2])

        if not self.axes_set:
            r = 1000;
            if 'cam2d' in mode:
                xroot, yroot, zroot = 0, 0, 0
            else:
                xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
            self.ax.set_xlim3d([-r + xroot, r + xroot])
            self.ax.set_zlim3d([-r + zroot, r + zroot])
            self.ax.set_ylim3d([-r + yroot, r + yroot])

            # self.ax.set_aspect('equal')
            # self.axes_set = True   # 固定点的画图
            self.axes_set = False   # 中心跟随的画图


def plot_gif_rlt(pose_gt, pose_recoverd, pose_mask, save_path):
    '''
    pose_gt: t j 3
    pose_recoverd:  t j 3
    pose_mask:  t j 1
    save_name: xxx.gif
    '''
    pose_gt = np.transpose(pose_gt[np.newaxis, ...], (0, 2, 1, 3)) * 1e3
    pose_recoverd = np.transpose(pose_recoverd[np.newaxis, ...], (0, 2, 1, 3)) * 1e3
    pose_mask = np.transpose(pose_mask[np.newaxis, ...], (0, 2, 1, 3))[0]
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plot_seq_gif(seqs=np.concatenate([pose_gt, pose_recoverd]),
                 labs=None,
                 data_set="Human36",
                 seq_masks=pose_mask,
                 extra_text=None,
                 save_path=save_path, figwidth=256 * 2, figheight=256)

def Wrap_plot_seq_gif(seqs, labs, save_path, figwidth=256 * 2, figheight=256):
    '''
    segs: n x t x j x 3
    labs: input_world, input_cam2d, input_cam3d, recoverd_world, RT3D_cam3d, RT2D_cam2d
    save_name: xxx.gif
    '''
    seqs = np.transpose(seqs, (0, 2, 1, 3)) * 1e3
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plot_seq_gif(seqs=seqs,
                 labs=labs,
                 data_set="Human36",
                 save_path=save_path, figwidth=figwidth, figheight=figheight)


def _zip_GIFplot_array(np_lst):
    """
    for plot function pre-preocess
    """
    lst = []
    for item in np_lst:
        if item.shape[-1] == 3:  # for 3D case
            lst.append(np.expand_dims(item, 0))
        elif item.shape[-1] == 2:
            tmp2d = np.expand_dims(item, 0)
            tmp2d = np.concatenate([tmp2d, np.zeros_like(tmp2d)[..., -1:]], axis=-1)
            lst.append(tmp2d)
        else:
            assert False, 'wrong data get'
    return np.concatenate(lst)

def save_2dpose_gif(pose2d_seq, save_name):
    # b j 2
    print('save gif: {}'.format(save_name))
    idx = np.arange(0, pose2d_seq.shape[0], 10)
    lables = ['input_cam2d']
    seqs = _zip_GIFplot_array([
        pose2d_seq[idx],
    ])
    gif_save_path = os.path.join('./results/gif', '{}.gif'.format(save_name))
    os.makedirs('./results/gif', exist_ok=True)
    Wrap_plot_seq_gif(seqs=seqs, labs=lables, save_path=gif_save_path)


def save_3dpose_gif(pose3d_seq, pose3d_seq2, save_name, downsample=5):
    # b j 3
    print('save gif: {}'.format(save_name))
    idx = np.arange(0, pose3d_seq.shape[0], downsample)
    lables = ['sktpose_world', 'sktposeRef_world']
    seqs = _zip_GIFplot_array([
        pose3d_seq[idx],
        pose3d_seq2[idx],
    ])
    # gif_save_path = os.path.join('./results/gif', '{}.gif'.format(save_name))
    # os.makedirs('./results/gif', exist_ok=True)
    Wrap_plot_seq_gif(seqs=seqs, labs=lables, save_path=save_name)




def plot_seq_gif(seqs, labs, data_set="Human36", seq_masks=None, extra_text=None, save_path=None, figwidth=768,
                 figheight=384, dpi=80):
    # kehong: change labs to string list
    import matplotlib
    if save_path is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    actions_l, njoints, body_members = select_dataset(data_set)

    if labs is not None:
        if len(labs) == seqs.shape[0]:
            labs_mode = "multi"
        else:
            assert labs.shape[0] == 4, \
                "seqs and labs len must match or be a single lab"
            labs_mode = "single"

    if seq_masks is not None:
        if len(seq_masks.shape) > 3 and seq_masks.shape[0] == seqs.shape[0]:
            mask_mode = "multi"
        else:
            assert seq_masks.shape[0] == njoints, \
                "seqs and labs len must match or be a single lab"
            mask_mode = "single"

    n_seqs = seqs.shape[0]
    # n_rows = np.int(np.ceil(np.sqrt(n_seqs) * figheight / figwidth))
    # n_cols = np.int(np.ceil(n_seqs / n_rows))
    n_rows = 3
    n_cols = n_seqs // 3 + 1
    figwidth = n_cols * 256
    figheight = n_rows * 256

    # create the index array to plot in cols-wise: re-order the index
    idex1 = np.arange(1, n_cols + 1)
    idex2 = [[item, item + 1 * n_cols, item + 2 * n_cols] for item in idex1]
    idex3 = np.array(idex2).reshape(-1)

    # fig = plt.figure(figsize=(figwidth / dpi, figheight / dpi), dpi=dpi)
    fig = plt.figure(figsize=(figwidth / dpi, figheight / dpi), dpi=dpi,
                     tight_layout={'pad': 0, 'h_pad': 0, 'w_pad': 0})

    # if labs is not None:
    #     title = 'Plotting samples from %s dataset' % data_set
    #     if labs_mode == "single":
    #         # seq_idx, subject, action, plen = labs
    #         # title += "\n action: %s  subject: %d  seq_idx: %d  length: %d" % \
    #         #           (actions_l[action], subject, seq_idx, plen)
    #         title += "\n {}".format(labs)
    #     fig.suptitle(title)

    axs = []
    obs = []
    view_dict = {
        'cam2d': [-90, -90],
        'cam3d': [-70, -89],
        'world': [30, -30]
    }
    for i in range(n_seqs):
        # ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        ax = fig.add_subplot(n_rows, n_cols, idex3[i], projection='3d')
        if labs is not None:
            view_key = labs[i].split('_')[-1]  # e.g., cam3d
            ax.view_init(elev=view_dict[view_key][0], azim=view_dict[view_key][1])
        else:
            ax.view_init(elev=30, azim=-30)
        ob = Ax3DPose(ax, data_set)
        axs.append(ax)
        obs.append(ob)

    seq_len = seqs.shape[2]
    frame_counter = fig.text(0.9 if labs is not None else 0.05, 0.1 if labs is not None else 0.95, 'frame: 0')
    if extra_text is not None:
        fig.text(0.1, 0.1, extra_text)

    # Tight layout often produces nice results
    # but requires the title to be spaced accordingly
    fig.tight_layout()
    fig.subplots_adjust(top=0.58)

    def update(frame):
        if frame >= seq_len:
            return
        for i in range(n_seqs):
            mask = None
            if seq_masks is not None:
                if mask_mode == "single" and i == 0:
                    mask = seq_masks[:, frame, 0]
                elif mask_mode == "multi":
                    mask = seq_masks[i, :, frame, 0]
            obs[i].update(seqs[i, :, frame, :], mask, labs[i])
            if labs is not None:
                if labs_mode == "multi":
                    # seq_idx, subject, action, plen = labs[i, ...]
                    # axs[i].set_xlabel('idx: %d \n act: %s' % (seq_idx, actions_l[action]))
                    # axs[i].set_title('label: {}'.format(labs[i]), fontsize=20, color='red')
                    axs[i].title.set_text('\n\nlabel: {}'.format(labs[i]))
                    # axs[i].text(.5, .9, s='label: {}'.format(labs[i]), color='red', fontweight='bold', fontsize=14,
                    #             horizontalalignment='center', transform=axs[i].transAxes)
            else:
                axs[i].set_xlabel('sequence %d' % i)
        frame_counter.set_text('frame: %d' % frame)
        frame_counter.set_color('red' if frame > seq_len // 2 else 'blue')

    anim = FuncAnimation(fig, update, frames=np.arange(0, seq_len + 10), interval=100, repeat_delay=1000)
    if save_path is not None:
        anim.save(save_path, dpi=dpi, writer='imagemagick')
    else:
        try:
            plt.show()
        except (KeyboardInterrupt, AttributeError):
            pass

    fig_size = (int(fig.get_figheight()), int(fig.get_figwidth()))
    plt.close(fig)

    return fig_size


# def plot_seq_pano(seqs, labs, data_set, seq_masks=None, extra_text=None, save_path=None, figwidth=768, figheight=768,
#                   dpi=80):
#     import matplotlib
#     if save_path is not None:
#         matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#
#     actions_l, njoints, body_members = select_dataset(data_set)
#
#     if labs is not None:
#         if labs.shape[0] == seqs.shape[0]:
#             labs_mode = "multi"
#         else:
#             assert labs.shape[0] == 4, \
#                 "seqs and labs len must match or be a single lab"
#             labs_mode = "single"
#
#     if seq_masks is not None:
#         if seq_masks.shape[0] == seqs.shape[0]:
#             mask_mode = "multi"
#         else:
#             assert seq_masks.shape[0] == njoints, \
#                 "seqs and labs len must match or be a single lab"
#             mask_mode = "single"
#
#     n_seqs = seqs.shape[0]
#     n_rows = np.int(np.ceil(np.sqrt(n_seqs) * figheight / figwidth))
#     n_cols = np.int(np.ceil(n_seqs / n_rows))
#
#     fig = plt.figure(figsize=(figwidth / dpi, figheight / dpi), dpi=dpi,
#                      tight_layout={'pad': 0, 'h_pad': 0, 'w_pad': 0})
#
#     if labs is not None:
#         title = 'Plotting samples from %s dataset' % data_set
#         if labs_mode == "single":
#             seq_idx, subject, action, plen = labs
#             title += "\n action: %s  subject: %d  seq_idx: %d  length: %d" % \
#                      (actions_l[action], subject, seq_idx, plen)
#         fig.suptitle(title)
#
#     axs = []
#     obs = []
#     for i in range(n_seqs):
#         ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
#         ax.view_init(elev=0, azim=-90)
#         ob = Ax3DPose(ax, data_set)
#         axs.append(ax)
#         obs.append(ob)
#
#     seq_len = seqs.shape[2]
#     if extra_text is not None:
#         fig.text(0.1, 0.1, extra_text)
#
#     pano_len = seq_len / 2
#     lcolor = "#3498db"
#     rcolor = "#e74c3c"
#     for i in range(n_seqs):
#         axs[i].set_xlabel("time")
#         axs[i].set_ylabel("y")
#         axs[i].set_zlabel("z")
#         r = 1000
#
#         yroot, zroot = 0, seqs[i, 0, 0, 2]
#         axs[i].set_xlim3d([0, r * pano_len])
#         axs[i].set_zlim3d([-r + zroot, r + zroot])
#         axs[i].set_ylim3d([-r + yroot, r + yroot])
#         # axs[i].set_aspect('equal')
#
#         mask = None
#         if seq_masks is not None:
#             if mask_mode == "single" and i == 0:
#                 mask = seq_masks[:, :, 0]
#             elif mask_mode == "multi":
#                 mask = seq_masks[i, :, :, 0]
#
#         for f in range(seq_len):
#             x_hip = seqs[i, 0, f, 0]
#             f_pos = (f / seq_len) * pano_len * r
#
#             for member in body_members.values():
#                 for j in range(len(member['joints']) - 1):
#                     j_idx_start = member['joints'][j]
#                     j_idx_end = member['joints'][j + 1]
#                     x = np.array([seqs[i, j_idx_start, f, 0] - x_hip + f_pos,
#                                   seqs[i, j_idx_end, f, 0] - x_hip + f_pos])
#                     y = np.array([seqs[i, j_idx_start, f, 1], seqs[i, j_idx_end, f, 1]])
#                     z = np.array([seqs[i, j_idx_start, f, 2], seqs[i, j_idx_end, f, 2]])
#                     axs[i].plot(x, y, z, lw=2, c=lcolor if member['side'] == 'left' else rcolor)
#
#             if mask is not None:
#                 for j in range(njoints):
#                     x = np.array([seqs[i, j, f, 0] - x_hip + f_pos])
#                     y = np.array([seqs[i, j, f, 1]])
#                     z = np.array([seqs[i, j, f, 2]])
#                     axs[i].plot(x, y, z, lw=2, c='black', markersize=8,
#                                 marker='o', linestyle='dashed',
#                                 visible=True if mask[j, f] == 0 else False)
#
#     if save_path is not None:
#         fig.savefig(save_path, dpi=dpi)
#     else:
#         try:
#             plt.show()
#         except (KeyboardInterrupt, AttributeError):
#             pass
#
#     fig_size = (int(fig.get_figheight()), int(fig.get_figwidth()))
#     plt.close(fig)
#
#     return fig_size
#
#
# def plot_seq_frozen(seqs, labs, data_set, seq_masks=None, extra_text=None, save_path=None, figwidth=768, figheight=768,
#                     dpi=80):
#     import matplotlib
#     if save_path is not None:
#         matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#
#     actions_l, njoints, body_members = select_dataset(data_set)
#
#     if labs is not None:
#         if labs.shape[0] == seqs.shape[0]:
#             labs_mode = "multi"
#         else:
#             assert labs.shape[0] == 4, \
#                 "seqs and labs len must match or be a single lab"
#             labs_mode = "single"
#
#     if seq_masks is not None:
#         if seq_masks.shape[0] == seqs.shape[0]:
#             mask_mode = "multi"
#         else:
#             assert seq_masks.shape[0] == njoints, \
#                 "seqs and labs len must match or be a single lab"
#             mask_mode = "single"
#
#     n_seqs = seqs.shape[0]
#     n_rows = np.int(np.ceil(np.sqrt(n_seqs) * figheight / figwidth))
#     n_cols = np.int(np.ceil(n_seqs / n_rows))
#
#     fig = plt.figure(figsize=(figwidth / dpi, figheight / dpi), dpi=dpi,
#                      tight_layout={'pad': 0, 'h_pad': 0, 'w_pad': 0})
#
#     # if labs is not None:
#     #     title = 'Plotting samples from %s dataset' % data_set
#     #     if labs_mode == "single":
#     #         seq_idx, subject, action, plen = labs
#     #         title += "\n action: %s  subject: %d  seq_idx: %d  length: %d" % \
#     #                   (actions_l[action], subject, seq_idx, plen)
#     #     fig.suptitle(title)
#
#     axs = []
#     obs = []
#     for i in range(n_seqs):
#         ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
#         ax.view_init(elev=30, azim=-30)
#         ob = Ax3DPose(ax, data_set)
#         axs.append(ax)
#         obs.append(ob)
#
#     seq_len = seqs.shape[2]
#     # if extra_text is not None:
#     #     fig.text(0.1, 0.1, extra_text)
#
#     lcolor = "#3498db"
#     rcolor = "#e74c3c"
#     for i in range(n_seqs):
#         axs[i].set_xlabel("")
#         axs[i].set_ylabel("")
#         axs[i].set_zlabel("")
#         axs[i].set_xticklabels([])
#         axs[i].set_yticklabels([])
#         axs[i].set_zticklabels([])
#         r = 1000
#         axs[i].set_xlim3d([np.mean(seqs[:, :, :, 0]) - r, np.mean(seqs[:, :, :, 0]) + r])
#         axs[i].set_ylim3d([np.mean(seqs[:, :, :, 1]) - r, np.mean(seqs[:, :, :, 1]) + r])
#         axs[i].set_zlim3d([np.mean(seqs[:, :, :, 2]) - r, np.mean(seqs[:, :, :, 2]) + r])
#         # axs[i].set_aspect('equal')
#
#         mask = None
#         if seq_masks is not None:
#             if mask_mode == "single" and i == 0:
#                 mask = seq_masks[:, :, 0]
#             elif mask_mode == "multi":
#                 mask = seq_masks[i, :, :, 0]
#
#         # def color_blend(f, color_a, color_b):
#         #     return ((color_b[0]/255 * f/seq_len) + (color_a[0]/255*(1-(f/seq_len))),
#         #             (color_b[1]/255 * f/seq_len) + (color_a[1]/255*(1-(f/seq_len))),
#         #             (color_b[2]/255 * f/seq_len) + (color_a[2]/255*(1-(f/seq_len))), 1.0)
#         cmap = matplotlib.cm.get_cmap('jet')
#
#         for f in range(seq_len):
#             for member in body_members.values():
#                 for j in range(len(member['joints']) - 1):
#                     j_idx_start = member['joints'][j]
#                     j_idx_end = member['joints'][j + 1]
#                     x = np.array([seqs[i, j_idx_start, f, 0], seqs[i, j_idx_end, f, 0]])
#                     y = np.array([seqs[i, j_idx_start, f, 1], seqs[i, j_idx_end, f, 1]])
#                     z = np.array([seqs[i, j_idx_start, f, 2], seqs[i, j_idx_end, f, 2]])
#                     lw = 2
#                     c = cmap(f / seq_len)
#                     if f == 0:
#                         c = "blue"
#                     elif f == (seq_len // 2):
#                         c = "black"
#                     elif f == seq_len - 1:
#                         c = "red"
#                     axs[i].plot(x, y, z, lw=lw, c=c)
#
#             # if mask is not None:
#             #     for j in range(njoints):
#             #         x = np.array([seqs[i, j, f, 0]])
#             #         y = np.array([seqs[i, j, f, 1]])
#             #         z = np.array([seqs[i, j, f, 2]])
#             #         axs[i].plot(x, y, z, lw=2, c='black', markersize=8,
#             #                     marker='o', linestyle='dashed',
#             #                     visible=True if mask[j, f] == 0 else False)
#
#     if save_path is not None:
#         fig.savefig(save_path, dpi=dpi)
#     else:
#         try:
#             plt.show()
#         except (KeyboardInterrupt, AttributeError):
#             pass
#
#     fig_size = (int(fig.get_figheight()), int(fig.get_figwidth()))
#     plt.close(fig)
#
#     return fig_size
#
#
# def plot_seq_emb(seq_emb, save_path):
#     seq_emb = (seq_emb - np.min(seq_emb)) / (np.max(seq_emb) - np.min(seq_emb))
#     seq_emb = (seq_emb * 255).astype(np.uint8)
#     im = Image.fromarray(seq_emb, mode='L')
#     im.save(save_path)
