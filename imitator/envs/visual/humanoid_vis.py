import numpy as np
from envs.common import mujoco_env


class HumanoidVisEnv(mujoco_env.MujocoEnv):
    def __init__(self, vis_model_file, nframes=6, focus=True):
        mujoco_env.MujocoEnv.__init__(self, vis_model_file, nframes)
        self.set_cam_first = set()
        self.focus = focus

    def step(self, a):
        return np.zeros((10, 1)), 0, False, dict()

    def reset_model(self):
        c = 0
        self.set_state(
            self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        )
        return None

    def sim_forward(self):
        self.sim.forward()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        if self.focus:
            self.viewer.cam.lookat[:2] = self.data.qpos[:2]
            self.viewer.cam.lookat[2] = 0.8
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 30
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.5
            self.viewer.cam.elevation = -10
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)
