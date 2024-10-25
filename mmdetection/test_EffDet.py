# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

def main():
    setup_cache_size_limit_of_dynamo()
    cfg = Config.fromfile('./projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco_trash.py')
    model_name = 'efficientdet_effb0_bifpn_8xb16_no_valid'
    cfg.work_dir = './work_dirs/'+model_name
    runner = RUNNERS.build(cfg)
    # start training
    runner.train()

if __name__ == '__main__':
    main()