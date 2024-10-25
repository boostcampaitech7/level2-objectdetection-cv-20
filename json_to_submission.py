import argparse
import json
import os
import pandas as pd
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description='Convert json file to submission csv.')
    parser.add_argument('json_file_path', help='Path to the JSON file with predictions.')
    parser.add_argument('output_path', help='Directory to save the output CSV file.')
    parser.add_argument('output_file', help='Name of the output CSV file.')
    parser.add_argument('score_threshold', type=float, help='Score threshold for filtering predictions.')
    parser.add_argument('--train', action='store_true', default=False, help='Enable train data to test.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open(args.json_file_path, 'r') as f:
        data = json.load(f)

    if args.train is True:
        original_json = '../dataset/train.json'
    else:
        original_json = '../dataset/test.json'
        
    coco = COCO(original_json)
    img_ids = coco.getImgIds()

    if args.train is True:
        file_names = [f'train/{str(img_id).zfill(4)}.jpg' for img_id in range(len(img_ids))]
    else:
        file_names = [f'test/{str(img_id).zfill(4)}.jpg' for img_id in range(len(img_ids))]
        
    prediction_strings = ['' for _ in range(len(img_ids))]

    for info in data:

        img_id = info['image_id']
        xmin, ymin, w, h = info['bbox']
        xmax, ymax = xmin+w, ymin+h
        score = info['score']
        cate_id = info['category_id']

        if score < args.score_threshold:
            continue

        prediction_list = map(str, [cate_id, score, xmin, ymin, xmax, ymax])
        prediction_string = ' '.join(prediction_list) + ' '

        prediction_strings[img_id] += prediction_string

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    os.makedirs(args.output_path, exist_ok=True)

    submission.to_csv(os.path.join(args.output_path, args.output_file+'.csv'), index=None)

    print('Create CSV file for submission!')

if __name__ == '__main__':
    main()