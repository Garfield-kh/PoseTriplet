import torch
import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(__file__))
from quaternion import qmul_np, qmul, qrot
from torch.utils.data import Dataset, DataLoader
from LaFan import LaFan1

class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None):
        assert len(offsets) == len(parents)
        
        self._offsets = torch.FloatTensor(offsets)
        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()
    
    def cuda(self):
        self._offsets = self._offsets.cuda()
        return self
    
    def num_joints(self):
        return self._offsets.shape[0]
    
    def offsets(self):
        return self._offsets
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()
        
    def forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                   self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
    
    def joints_left(self):
        return self._joints_left
    
    def joints_right(self):
        return self._joints_right
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

if __name__=="__main__":
    skeleton_mocap = Skeleton(offsets=[
       [-42.198200,91.614723,-40.067841],
       [ 0.103456,1.857829,10.548506],
       [43.499992,-0.000038,-0.000002],
       [42.372192,0.000015,-0.000007],
       [ 17.299999,-0.000002,0.000003],
       [0.000000,0.000000,0.000000],

       [0.103457,1.857829,-10.548503],
       [43.500042,-0.000027,0.000008],
       [42.372257,-0.000008,0.000014],
       [17.299992,-0.000005,0.000004],
       [0.000000,0.000000,0.000000],

       [6.901968,-2.603733,-0.000001],
       [12.588099,0.000002,0.000000],
       [12.343206,0.000000,-0.000001],
       [25.832886,-0.000004,0.000003],
       [11.766620,0.000005,-0.000001],
       [0.000000,0.000000,0.000000],

       [19.745899,-1.480370,6.000108],
       [11.284125,-0.000009,-0.000018],
       [33.000050,0.000004,0.000032],
       [25.200008,0.000015,0.000008],
       [0.000000,0.000000,0.000000],

       [19.746099,-1.480375,-6.000073],
       [11.284138,-0.000015,-0.000012],
       [33.000092,0.000017,0.000013],
       [25.199780,0.000135,0.000422],
       [0.000000,0.000000,0.000000]
    ],
    parents=[-1,  0,  1,  2,  3,  4,\
              0,  6,  7,  8,  9,\
              0, 11, 12, 13, 14, 15,\
              13, 17, 18, 19, 20, 
              13, 22, 23, 24, 25])

    skeleton_mocap.remove_joints([5,10,16,21,26])


    # lafan_data = LaFan1('D:\\ubisoft-laforge-animation-dataset\\lafan1\\lafan1', train = False, debug=False)
    lafan_data = LaFan1('./lafan1/lafan1', train = False, debug=True)
    lafan_loader = DataLoader(lafan_data, batch_size=32, shuffle=False, num_workers=4)
    for i_batch, sample_batched in enumerate(lafan_loader):
        pos_batch = skeleton_mocap.forward_kinematics(sample_batched['local_q'], sample_batched['root_p'])
        # print(pos_batch[0,:,0].cpu().numpy())
        # break
        