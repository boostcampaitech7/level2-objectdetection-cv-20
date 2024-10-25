import cv2
import os, random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotation, data_dir, transforms=None, mosaic=False):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms
        self.mosaic = mosaic

    def __getitem__(self, index: int):
        return self.load_mosaic(index) if self.mosaic else self.load_image_target(index)

    def load_image_info(self, index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.data_dir, image_info['file_name'])    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns])
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        labels = np.array([x['category_id'] + 1 for x in anns], dtype=np.int64)

        return image, boxes, labels, image_id

    def load_image_target(self, index):
        image, boxes, labels, image_id = self.load_image_info(index)
        target = {
            'boxes': boxes,
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([index])
        }

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, image_id

    def load_mosaic(self, index):
        # Size of Mosaic Image
        s = 1024
         
        indices = [index] + [random.randint(0, len(self.coco.getImgIds())-1) for _ in range(3)]
        final_boxes, final_labels = [], []
        mosaic_img = np.zeros((s, s, 3), dtype=np.float32)
        xc, yc = np.random.randint(s * 0.25, s * 0.75, (2,))

        for i, idx in enumerate(indices):
            image, boxes, labels, _ = self.load_image_info(idx)
            h, w = image.shape[:2]

            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
                x1b, y1b, x2b, y2b = s - xc, s - yc, s, s
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, 0, s, yc
                x1b, y1b, x2b, y2b = 0, s - yc, s - xc, s
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = 0, yc, xc, s
                x1b, y1b, x2b, y2b = s - xc, 0, s, s-yc
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, s, s
                x1b, y1b, x2b, y2b = 0, 0, s-xc, s-yc

            offset_x, offset_y  = x1a - x1b, y1a - y1b
            boxes[:, 0] += offset_x
            boxes[:, 1] += offset_y
            boxes[:, 2] += offset_x
            boxes[:, 3] += offset_y

            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            final_boxes.append(boxes)
            final_labels.append(labels)

        final_boxes = np.vstack(final_boxes)
        final_boxes[:, 0:] = np.clip(final_boxes[:, 0:], 0, s).astype(np.int32)
        final_labels = np.concatenate(final_labels, axis=0)

        keep = np.where((final_boxes[:, 2] > final_boxes[:, 0]) & (final_boxes[:, 3] > final_boxes[:, 1]))[0]
        final_boxes, final_labels = final_boxes[keep], final_labels[keep]

        target = {
            'boxes': final_boxes,
            'labels': torch.tensor(final_labels, dtype=torch.int64),
            'image_id': torch.tensor([index])
        }

        transform = A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        sample = {
            'image' : mosaic_img, 
            'bboxes': final_boxes,
            'labels': final_labels
        }
        mosaic_img = transform(**sample)['image']
        target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return mosaic_img, target, index

    def __len__(self) -> int:
        return len(self.coco.getImgIds())