from core import TrajBatch
import numpy as np


class TrajBatchEgo(TrajBatch):

    def __init__(self, memory_list):
        super().__init__(memory_list)
        self.v_metas = np.stack(next(self.batch))

