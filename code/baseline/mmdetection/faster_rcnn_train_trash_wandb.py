#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[6]:


# 명령줄 인자 파서 설정
parser = argparse.ArgumentParser(description="Model Training Script")
parser.add_argument('--pos', type=float, default=0.5, help='pos_iou_thr')
parser.add_argument('--neg', type=float, default=0.5, help='neg_iou_thr')
parser.add_argument('--min', type=float, default=0.3, help='min_pos_iou')
parser.add_argument('--score', type=float, default=0.1, help='score_thr')
args = parser.parse_args()


# In[2]:


print(f"pos_iou_thr: {args.pos}")
print(f"neg_iou_thr: {args.neg}")
print(f"min_pos_thr: {args.min}")
print(f"score_thr: {args.score}")


# In[8]:


model_name = f"faster_rcnn_fpn_pos_iou_thr{args.pos}_neg_iou_thr{args.neg}_min_pos_iou{args.min}_score_thr{args.score}"


# In[9]:


classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
#cfg = Config.fromfile('./configs/a_my_config/my_config.py')
cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
root='../../../../../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
#cfg.data.train.palette = palette

cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'split_train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + 'split_val.json' # test json 정보
cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 2

cfg.seed = 20
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/'+ model_name

cfg.model.roi_head.bbox_head.num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

cfg.runner.max_epochs = 30
cfg.workflow = [('train', 1)]
cfg.runner.meta = dict(exp_name = "exp_name")


# In[ ]:


cfg.model.train_cfg.rpn.assigner.pos_iou_thr = args.pos
cfg.model.train_cfg.rpn.assigner.neg_iou_thr = args.neg
cfg.model.train_cfg.rpn.assigner.min_pos_iou = args.min
cfg.model.train_cfg.rcnn.assigner.pos_iou_thr = args.pos
cfg.model.train_cfg.rcnn.assigner.neg_iou_thr = args.neg
cfg.model.train_cfg.rcnn.assigner.min_pos_iou =args.neg
cfg.model.test_cfg.rcnn.score_thr = args.score


# In[11]:


print(cfg.model.train_cfg.rpn)
print(cfg.model.train_cfg.rcnn)
print(cfg.model.test_cfg.rcnn)


# In[12]:


wandb.login()

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


# In[13]:


# build_dataset
datasets = [build_dataset(cfg.data.train)]
#valid_datasets = []


# In[14]:


# dataset 확인
datasets[0]


# In[15]:


# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()


# In[16]:


cfg.runner.max_epochs = 20


# In[17]:


# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)


# In[ ]:




