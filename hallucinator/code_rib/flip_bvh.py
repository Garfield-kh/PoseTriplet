import numpy as np
import os 
from quaternion import euler_to_quaternion, qeuler_np

root = './lafan1/lafan1'
def flip_bvh(filename):
    fout = open(os.path.join(root, filename.replace('.bvh', '_flip.bvh')), 'w')
    cnt = 0
    for line in open(os.path.join(root, filename), 'r'):
        cnt += 1
        if cnt <= 134:
            fout.write(line)
        else:
            line = line.split('\n')[0].split(' ')[:69]
            line = np.reshape(np.array([float(x) for x in line]), [23, 3])
            line[0,2] *= -1.0
            
            quat = euler_to_quaternion(line[1:] / 180.0 * np.pi, 'zyx')
            quat[:,0] *= -1.0
            quat[:,1] *= -1.0
            line[1:] = qeuler_np(quat, 'zyx') / np.pi * 180.0
            
            left_idx = [2,3,4,5,15,16,17,18]
            right_idx = [6,7,8,9,19,20,21,22]
            line[left_idx+right_idx] = line[right_idx+left_idx].copy()
            
            line = np.reshape(line, (69,))
            new_line = ''
            for s in line[:-1]:
                new_line += (str(s) + ' ')
            new_line += (str(line[-1]) + '\n')
            fout.write(new_line)
for filename in os.listdir(root):
    flip_bvh(filename)
    # assert 0
    print(filename)