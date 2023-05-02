from omegaconf import OmegaConf

from models.mvs4net_utils import FPN4

def test_fpn_basic():
    cfg = OmegaConf.load("config.yaml")

    