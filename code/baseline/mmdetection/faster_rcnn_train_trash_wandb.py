#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

import argparse
import wandb


# In[4]:


from itertools import product

# 설정할 수 있는 모든 조합을 생성합니다.
combinations = list(product([0.9, 0.5], [0.5, 0.0], [0.3, 0.5], [0.1, 0.7]))

# 각 조합에 대해 모델을 훈련합니다.
for pos_iou_thr, neg_iou_thr, min_pos_iou, score_thr in combinations:
    # 클래스 정의
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # 설정 파일 로드
    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    root='../../../../../dataset/'

    # 데이터셋 설정 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train_80.json' 
    cfg.data.train.pipeline[2]['img_scale'] = (512,512) 

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + 'val_20.json' 
    cfg.data.val.pipeline[1]['img_scale'] = (512,512) 

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' 
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) 
    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 2

    cfg.seed = 20
    cfg.gpu_ids = [0]

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    cfg.runner.max_epochs = 20
    # 모델 설정
    cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.model.train_cfg.rpn.assigner.pos_iou_thr = pos_iou_thr
    cfg.model.train_cfg.rpn.assigner.neg_iou_thr = neg_iou_thr
    cfg.model.train_cfg.rpn.assigner.min_pos_iou = min_pos_iou
    cfg.model.train_cfg.rcnn.assigner.pos_iou_thr = pos_iou_thr
    cfg.model.train_cfg.rcnn.assigner.neg_iou_thr = neg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.min_pos_iou = min_pos_iou
    cfg.model.test_cfg.rcnn.score_thr = score_thr

    # 모델 이름 생성
    model_name = f"faster_rcnn_fpn_pos_iou_thr{pos_iou_thr}_neg_iou_thr{neg_iou_thr}_min_pos_iou{min_pos_iou}_score_thr{score_thr}"
    cfg.work_dir = './work_dirs/'+ model_name

    # 로그 설정
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MyMMDetWandbHook',
                init_kwargs={'project' : 'object-detection',
                            'name' : model_name},
                interval=10,
                log_checkpoint=True,
                log_checkpoint_metadata=True,
                num_eval_images=100
                )
                ]

    # 모델 훈련
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.init_weights()
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
    print(f"모델 {model_name} 훈련 완료!")

