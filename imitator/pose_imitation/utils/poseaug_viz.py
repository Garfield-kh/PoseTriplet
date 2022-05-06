
"""
Functions to visualize human poses
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def show3DposePair(realt3d, faket3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=True, pred=False):  # blue, orange
  """
  Visualize a 3d skeleton pair

  Args
  channels: 96x1 vector. The pose to plot.
  ax: matplotlib 3d axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  realt3d = np.reshape(realt3d, (16, -1))
  faket3d = np.reshape(faket3d, (16, -1))

  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  for idx, vals in enumerate([realt3d, faket3d]):
    # Make connection matrix
    for i in np.arange(len(I)):
      x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
      if idx == 0:
        ax.plot(x, z, -y, lw=2, c='k')
      #        ax.plot(x,y, z,  lw=2, c='k')

      elif idx == 1:
        ax.plot(x, z, -y, lw=2, c='r')
      #        ax.plot(x,y, z,  lw=2, c='r')

      else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 1  # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

  # Get rid of the ticks and tick labels
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #  ax.set_zticks([])
  #
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])
  #  ax.set_zticklabels([])
  #     ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
               gt=False,pred=False): # blue, orange
    """
    Visualize a 3d skeleton

    Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
    Returns
    Nothing. Draws on ax.
    """

    #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (16, -1) )

    I  = np.array([0,1,2,0,4,5,0,7,8,8,10,11,8,13,14]) # start points
    J  = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        if gt:
            ax.plot(x,z, -y,  lw=2, c='k')
        #        ax.plot(x,y, z,  lw=2, c='k')

        elif pred:
            ax.plot(x,z, -y,  lw=2, c='r')
        #        ax.plot(x,y, z,  lw=2, c='r')

        else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
            ax.plot(x, z, -y,  lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 1 # space around the subject
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])


    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("-y")

    # Get rid of the ticks and tick labels
    #  ax.set_xticks([])
    #  ax.set_yticks([])
    #  ax.set_zticks([])
    #
    #  ax.get_xaxis().set_ticklabels([])
    #  ax.get_yaxis().set_ticklabels([])
    #  ax.set_zticklabels([])
#     ax.set_aspect('equal')

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True):
  """
  Visualize a 2d skeleton

  Args
  channels: 64x1 vector. The pose to plot.
  ax: matplotlib axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  vals = np.reshape(channels, (-1, 2))
  # plt.plot(vals[:,0], vals[:,1], 'ro')
  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange(len(I)):
    x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
    #         print('x',x)
    #         print(y)
    ax.plot(x, -y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #
  #  # Get rid of tick labels
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])

  RADIUS = 1  # space around the subject
  xroot, yroot = vals[0, 0], vals[0, 1]
  #     ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  #     ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])

  ax.set_xlim([-1, 1])
  ax.set_ylim([-1, 1])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("-y")

  ax.set_aspect('equal')


##############################
# wrap for simple usage
##############################
def wrap_show3d_pose(vals3d):
    fig3d = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d = Axes3D(fig3d)
    show3Dpose(vals3d, ax3d)
    plt.show()


def wrap_show2d_pose(vals2d):
    ax2d = plt.axes()
    show2Dpose(vals2d, ax2d)
    plt.show()

import os

import matplotlib.pyplot as plt

# from common.camera import project_to_2d
# from common.viz import show3Dpose, show3DposePair, show2Dpose
import torch
h36m_cam3d_sample = torch.from_numpy(np.load('tmp/h36m_sample/inputs_3d_cam_f32.npy'))
h36m_cam2d_sample = torch.from_numpy(np.load('tmp/h36m_sample/inputs_2d_cam_f32.npy'))



def plot_poseaug(Gcam_rlt, e_2dpose, iter, log_dir):
    num = Gcam_rlt['pose3D_camed'].shape[0]
    idx1 = np.random.randint(num)
    inputs_3d = Gcam_rlt['pose3D_camed'][0]
    outputs_3d_ba = Gcam_rlt['pose3D_camed'][idx1]
    h36m_cam3d_sample_0 = h36m_cam3d_sample[0,0]
    h36m_cam3d_sample_random = h36m_cam3d_sample[np.random.randint(5000),np.random.randint(32)]

    inputs_2d = Gcam_rlt['pose2D_camed'][0]
    outputs_2d_ba = Gcam_rlt['pose2D_camed'][idx1]
    h36m_cam2d_sample_0 = h36m_cam2d_sample[0,0]
    e_2dpose = e_2dpose[0]

    # plot the augmented pose from origin -> ba -> bl -> rt
    _plot_poseaug(
        inputs_3d.cpu().detach().numpy(), inputs_2d.cpu().detach().numpy(),
        outputs_3d_ba.cpu().detach().numpy(), outputs_2d_ba.cpu().detach().numpy(),
        h36m_cam3d_sample_0.cpu().detach().numpy(), h36m_cam2d_sample_0.cpu().detach().numpy(),
        h36m_cam3d_sample_random.cpu().detach().numpy(), e_2dpose.cpu().detach().numpy(),
        iter, log_dir
    )



def _plot_poseaug(
        tmp_inputs_3d, tmp_inputs_2d,
        tmp_outputs_3d_ba, tmp_outputs_2d_ba,
        tmp_h36m_cam3d_sample_0, tmp_h36m_cam2d_sample_0,
        tmp_h36m_cam3d_sample_random, tmp_e_2dpose,
        iter, log_dir
):
    # plot all the rlt
    fig3d = plt.figure(figsize=(16, 8))

    # input 3d
    ax3din = fig3d.add_subplot(2, 4, 1, projection='3d')
    ax3din.set_title('input 3D')
    show3Dpose(tmp_inputs_3d, ax3din, gt=False)

    # show source 2d
    ax2din = fig3d.add_subplot(2, 4, 5)
    ax2din.set_title('input 2d')
    show2Dpose(tmp_inputs_2d, ax2din)

    # input 3d to modify 3d
    ax3dba = fig3d.add_subplot(2, 4, 2, projection='3d')
    ax3dba.set_title('input/ba 3d')
    show3Dpose(tmp_outputs_3d_ba, ax3dba)

    # show source 2d
    ax2dba = fig3d.add_subplot(2, 4, 6)
    ax2dba.set_title('ba 2d')
    show2Dpose(tmp_outputs_2d_ba, ax2dba)

    # input 3d to modify 3d
    ax3dbl = fig3d.add_subplot(2, 4, 3, projection='3d')
    ax3dbl.set_title('tmp_h36m_cam3d_sample_0')
    show3Dpose(tmp_h36m_cam3d_sample_0, ax3dbl)

    # show source 2d
    ax2dbl = fig3d.add_subplot(2, 4, 7)
    ax2dbl.set_title('tmp_h36m_cam2d_sample_0')
    show2Dpose(tmp_h36m_cam2d_sample_0, ax2dbl)

    # modify 3d to rotated 3d
    ax3drt = fig3d.add_subplot(2, 4, 4, projection='3d')
    ax3drt.set_title('tmp_h36m_cam3d_sample_random')
    show3Dpose(tmp_h36m_cam3d_sample_random, ax3drt, gt=False)

    # rt 3d to 2d
    ax2d = fig3d.add_subplot(2, 4, 8)
    ax2d.set_title('tmp_e_2dpose')
    show2Dpose(tmp_e_2dpose, ax2d)

    os.makedirs('{}/tmp_viz'.format(log_dir), exist_ok=True)
    image_name = '{}/tmp_viz/iter_{:0>4d}.png'.format(log_dir, iter)
    plt.savefig(image_name)
    plt.close('all')

