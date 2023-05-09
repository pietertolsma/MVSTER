from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from models.mvs4net_utils import FPN4
from datasets.totemvs import MVSDataset


def test_fpn_basic():
    cfg = OmegaConf.load("config.yaml")

    fpn = FPN4(cfg)
    data = MVSDataset(cfg, "train")
    dl = DataLoader(data, 1, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=False)

    item = next(iter(dl))
    x = item["imgs"][0][0].unsqueeze(0)

    outputs = fpn(x)

    assert len(outputs) == 4

    cfg.fpn.levels = [0,1]
    fpn2 = FPN4(cfg)

    item = next(iter(dl))
    x = item["imgs"][0][0].unsqueeze(0)

    outputs = fpn2(x)

    assert len(outputs) == 2