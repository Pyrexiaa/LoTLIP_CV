import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
import random
import open_clip
import numpy as np
import pandas as pd
from ultralytics import YOLO

from utils import (
    convert_bbox,
    draw_bounding_box,
    calculate_metrics,
    group_bbox_by_filename, 
    plot_similarity_histogram,
    match_predictions_to_ground_truth,
)

TP_COLOR = (0, 255, 0)  # GREEN
FP_COLOR = (0, 0, 255)  # BLUE
FN_COLOR = (255, 0, 0)  # RED


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Path to model weights")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Path to output")
    parser.add_argument("--data", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--sim-thres", type=float, default=0., help="Threshold for cosine similarity")
    parser.add_argument("--iou-thres", type=float, default=0.75, help="Threshold for bbox IoU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # multi-GPU setups
    
    # create output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    SAVE_IMG_PATH = str(Path(args.output_dir, 'images_labelled'))
    if not os.path.exists(SAVE_IMG_PATH):
        os.makedirs(SAVE_IMG_PATH)
    
    
    # model config
    LOTLIP_MODEL_ARCHI   = "lotlip_bert-ViT-B-16"
    LOTLIP_MODEL_WEIGHTS = "model.pt"
    YOLO_MODEL_WEIGHTS   = "yolov8n.pt"
    
    # dataset path
    IMAGES_DIR     = Path(args.data, "images_raw")
    QUERY_CSV_PATH = Path(args.data, "query.csv")
    BBOX_GT_PATH   = Path(args.data, "bbox_selected.json")
    
    # load dataset
    QUERIES = pd.read_csv(QUERY_CSV_PATH)
    num_samples = QUERIES.shape[0]
    
    with open(BBOX_GT_PATH, 'r') as f:
        BBOX_GT = group_bbox_by_filename(json.load(f))
    
    
    # load LOTLIP model
    lotlip_model, _, preprocess = open_clip.create_model_and_transforms(
        LOTLIP_MODEL_ARCHI,
        pretrained=LOTLIP_MODEL_WEIGHTS,
        cache_dir=args.cache_dir,
    )
    lotlip_model.eval()
    tokenizer = open_clip.get_tokenizer(LOTLIP_MODEL_ARCHI)
    
    # load YOLOv8 model
    yolo_model = YOLO(YOLO_MODEL_WEIGHTS)
    
    
    # main inferencing loop
    sim_scores = []
    image_files = []
    selected_bboxes = {}
    for i in tqdm(range(num_samples)):
        image_file = str(QUERIES.loc[i, 'image'])
        query      = str(QUERIES.loc[i, 'query'])
        num_bbox   = int(QUERIES.loc[i, 'num_bbox'])
        
        image_path = str(Path(IMAGES_DIR, image_file))
        image_PIL = Image.open(image_path)
        img_width, img_height = image_PIL.size
        image_files.append(image_file)
        
        bbox_gt = BBOX_GT.get(image_file)
        assert len(bbox_gt) == num_bbox, image_file
        
        # run object detection
        results = yolo_model(image_path)

        output = []
        image_crops = []
        for result in results:
            for box in result.boxes.data.tolist():
                x_min, y_min, x_max, y_max, confidence, class_id = box
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width    = (x_max - x_min) / img_width
                height   = (y_max - y_min) / img_height
                
                # save bbox
                output.append({
                    "image": image_file,
                    "bbox": [
                        x_center, 
                        y_center, 
                        width, 
                        height
                    ],
                    # "bbox_raw": [
                    #     x_min,
                    #     y_min,
                    #     x_max,
                    #     y_max
                    # ],
                    "class_id": int(class_id),
                    "confidence": confidence,
                })
                
                # save cropped bbox image
                image_crops.append(
                    image_PIL.crop((
                        x_min,
                        y_min,
                        x_max,
                        y_max
                    ))
                )
        
        if len(image_crops) > 0:
            # apply default image preprocessing
            image_crops = [preprocess(img.convert("RGB")) for img in image_crops]
            image_crops_tensor = torch.stack(image_crops)   # Shape: [N, 3, H, W]
            
            # tokenize query text
            query_tokens = tokenizer([query])
            
            # embed both query text & cropped images
            with torch.no_grad():
                text_embedding = lotlip_model.encode_text(query_tokens)           # Shape: [1, D]
                image_embeddings = lotlip_model.encode_image(image_crops_tensor)  # Shape: [N, D]
            
            # compute cosine similarities, value range: [-1, 1]
            similarities = torch.nn.functional.cosine_similarity(text_embedding, image_embeddings) # Shape: [N]
            similarities = similarities.tolist()
            
            # save for visualisation
            sim_scores += similarities
            
            # select most relevant bbox
            relevant_bboxes = []
            for j, similarity in enumerate(similarities):
                if similarity > args.sim_thres:
                    bbox = output[j]
                    bbox["similarity"] = float(similarity)
                    relevant_bboxes.append(bbox)
            
            selected_bboxes[image_file] = relevant_bboxes
        else:
            # yolo detected nothing
            selected_bboxes[image_file] = []
        
        # save selected bbox
        with open(Path(args.output_dir, 'selected_bbox.json'), 'w') as f:
            json.dump(selected_bboxes, f, indent=4)
    
    
    # visualise distribution of semantic similarity scores
    plot_similarity_histogram(sim_scores, str(Path(args.output_dir, "hist_similarities.png")), log_scale=True)
    
    
    # compute metrics
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    processed_bbox = {}
    
    for image_file in tqdm(image_files):
        
        bbox_pred = selected_bboxes[image_file]
        bbox_gt   = BBOX_GT[image_file]
        
        image_path = str(Path(IMAGES_DIR, image_file))
        image_PIL = Image.open(image_path)
        img_width, img_height = image_PIL.size
        
        # categorize predictions into TP, FP, FN based on class_id and IoU
        TP, FP, FN = match_predictions_to_ground_truth(bbox_pred, bbox_gt, img_height, img_width, args.iou_thres)
        
        metrics = calculate_metrics(len(TP), len(FP), len(FN))
        
        # accumulate metrics
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1s.append(metrics['f1'])
        
        processed_bbox[image_file] = {
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "TP": [k[0] for k in TP],  # pred
            "FP": [k[0] for k in FP],  # pred
            "FN": [k[1] for k in FN],  # gt
        }
        
        # save results
        with open(Path(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(processed_bbox, f, indent=4)
        
        # label bbox on image
        for tp in TP:
            bbox     = tp[0]['bbox']
            class_id = tp[0]['class_id']
            sim      = tp[0].get('similarity', 0)
            
            bbox_convertted = convert_bbox(bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height)
            image_PIL = draw_bounding_box(
                image_PIL,
                bbox_convertted[0],
                bbox_convertted[1],
                bbox_convertted[2],
                bbox_convertted[3],
                class_id,
                sim,
                TP_COLOR
            )
        for fp in FP:
            bbox     = fp[0]['bbox']
            class_id = fp[0]['class_id']
            sim      = fp[0].get('similarity', 0)
            
            bbox_convertted = convert_bbox(bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height)
            image_PIL = draw_bounding_box(
                image_PIL,
                bbox_convertted[0],
                bbox_convertted[1],
                bbox_convertted[2],
                bbox_convertted[3],
                class_id,
                sim,
                FP_COLOR
            )
        for fn in FN:
            bbox     = fn[1]['bbox']
            class_id = fn[1]['class_id']
            sim      = fn[1].get('similarity', 0)
            
            bbox_convertted = convert_bbox(bbox[0], bbox[1], bbox[2], bbox[3], img_width, img_height)
            image_PIL = draw_bounding_box(
                image_PIL,
                bbox_convertted[0],
                bbox_convertted[1],
                bbox_convertted[2],
                bbox_convertted[3],
                class_id,
                sim,
                FN_COLOR
            )
        image_PIL.save(str(Path(SAVE_IMG_PATH, image_file)))
    
    
    # compute mean metrics
    mean_accuracy = sum(accuracies) / len(accuracies)
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mean_f1 = sum(f1s) / len(f1s)
    
    # save mean metrics
    with open(Path(args.output_dir, 'mean_metrics.json'), 'w') as f:
        json.dump(
            {
                "num_samples": len(accuracies),
                "mean_accuracy": mean_accuracy,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_f1": mean_f1
            },
            f,
            indent=4
        )

if __name__ == "__main__":
    main()
