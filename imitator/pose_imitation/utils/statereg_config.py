import yaml
import os
from utils import recreate_dirs


class Config:

    def __init__(self, cfg_id, create_dirs=False):
        self.id = cfg_id
        cfg_name = 'config/statereg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'results'
        self.cfg_dir = '%s/statereg/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if create_dirs:
            recreate_dirs(self.log_dir, self.tb_dir)

        # training config
        self.meta_id = cfg['meta_id']
        self.seed = cfg['seed']
        self.fr_num = cfg['fr_num']
        self.v_net = cfg.get('v_net', 'lstm')
        self.v_net_param = cfg.get('v_net_param', None)
        self.v_hdim = cfg['v_hdim']
        self.mlp_dim = cfg['mlp_dim']
        self.cnn_fdim = cfg['cnn_fdim']
        self.lr = cfg['lr']
        self.num_epoch = cfg['num_epoch']
        self.iter_method = cfg['iter_method']
        self.shuffle = cfg.get('shuffle', False)
        self.num_sample = cfg.get('num_sample', 20000)
        self.save_model_interval = cfg['save_model_interval']
        self.fr_margin = cfg['fr_margin']
        self.pose_only = cfg.get('pose_only', False)
        self.causal = cfg.get('causal', False)
        self.cnn_type = cfg.get('cnn_type', 'mlp')

        # misc config
        self.humanoid_model = cfg['humanoid_model']
        self.vis_model = cfg['vis_model']
