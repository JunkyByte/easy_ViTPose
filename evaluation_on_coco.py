# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

import cv2
from easy_ViTPose.inference import VitInference
from pathlib import Path
import os
from tqdm.auto import tqdm

from pycocotools.coco import COCO  
from pycocotools.cocoeval import COCOeval  
from statistics import mean
import json
import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Argument Parser for infer')
    parser.add_argument('--model_path', type=str, 
                        help='Path to the ViT Pose model')
    parser.add_argument('--model-name', type=str, choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo_path', type=str,
                        help='Path to the YOLOv8 model')
    parser.add_argument('--img_folder_path', type=str, 
                        help='Path to the folder containing images')
    parser.add_argument('--annFile', type=str,
                        help='Path to the COCO annotations file')
    return parser.parse_args()


def evaluation_on_coco(model_path, model_name, yolo_path, img_folder_path, annFile):
    # get image IDs of images in val set 
    # Opening JSON file
    f = open(annFile)
    gt_annotations = json.load(f)
    f.close()

    image_ids = set()
    for ann in gt_annotations['images']:
        image_ids.add(ann['id'])
    

    model = VitInference(model_path, yolo_path, model_name = model_name, yolo_size=640, is_video=False, device=None)
    results_list = []

    for image_id in tqdm(image_ids):
        # run inference here
        img_path = os.path.join(img_folder_path, str(image_id).zfill(12) + '.jpg')
        img = cv2.imread(img_path)

        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_keypoints = model.inference(img)
        for key in frame_keypoints:
            results_element = {}
            results_element['image_id'] = image_id
            results_element['category_id'] = 1
            results_element['score'] = model._scores_bbox[key]
            results_element['bbox'] = []
            keypoints = []
            for k in frame_keypoints[key]:
                keypoints.append(float(round(k[1], 0)))
                keypoints.append(float(round(k[0], 0)))
                keypoints.append(0)
            results_element['keypoints'] = keypoints
            results_list.append(results_element)


    # Define the file path where you want to save the JSON file
    file_path = "results.json"
    # Save the list of dictionaries to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(results_list, json_file, indent=4)


    #initialize COCO ground truth api
    annType = 'keypoints'    
    cocoGt=COCO(annFile)
    #initialize COCO detections api
    resFile="results.json"
    cocoDt=cocoGt.loadRes(resFile)
    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = [int(i) for i in image_ids]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = parse_arguments()
    evaluation_on_coco(args.model_path, args.model_name, args.yolo_path, args.img_folder_path, args.annFile)