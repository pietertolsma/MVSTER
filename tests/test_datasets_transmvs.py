from omegaconf import OmegaConf
import numpy as np

from datasets.transmvs import MVSDataset

def test_basic():
    cfg = OmegaConf.load("config.yaml")
    data  = MVSDataset(cfg)

    assert len(data) == 24

def test_custom_fpn():
    cfg = OmegaConf.load("config.yaml")

    cfg.fpn.levels = [2, 3]

    data  = MVSDataset(cfg)

    entry = data[0]
    assert len(entry["proj_matrices"]) == 2

    assert entry["proj_matrices"]["stage1"].shape == (6, 2, 4, 4)
    assert entry["proj_matrices"]["stage2"].shape == (6, 2, 4, 4)

    assert (2*entry["proj_matrices"]["stage1"][:, 0, :3, 3] == \
    entry["proj_matrices"]["stage2"][:, 0, :3, 3]).all()

    assert len(entry["depth"]) == 2
    assert np.all(2*np.array(entry["depth"]["stage1"].shape) == \
                  np.array(entry["depth"]["stage2"].shape))