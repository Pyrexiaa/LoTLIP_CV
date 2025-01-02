import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def get_bbox(
    ground_truths: list[dict],
    filename: str
) -> list[dict]:
    """Selects the corresponding bboxes given the image filename"""
    
    selected_bbox = []
    for d in ground_truths:
        if d.get("image") == filename:
            selected_bbox.append(d)
    return selected_bbox


def group_bbox_by_filename(
    ground_truths: list[dict]
) -> dict[str, list[dict]]:
    """Groups a list of bbox according to filename"""
    
    unique_filenames = set([b["image"] for b in ground_truths])
    
    grouped_bbox = {}
    for filename in unique_filenames:
        grouped_bbox[filename] = get_bbox(ground_truths, filename)
    
    return grouped_bbox


def convert_bbox(
    bbox_x_center: float,
    bbox_y_center: float,
    bbox_width: float,
    bbox_height: float,
    img_width: float,
    img_height: float,
) -> list[float]:
    """Converts a bbox from (center, bbox_size) format to (x,y) format"""
    
    x_min = (bbox_x_center * img_width) - (bbox_width * img_width) / 2
    x_max = (bbox_x_center * img_width) + (bbox_width * img_width) / 2
    y_min = (bbox_y_center * img_height) - (bbox_height * img_height) / 2
    y_max = (bbox_y_center * img_height) + (bbox_height * img_height) / 2
    
    return [x_min, y_min, x_max, y_max]


def compute_iou(
    box1: list[float],
    box2: list[float]
) -> float:
    """Computes IoU between 2 bboxes"""
    
    # box1, box2 in format [x_min, y_min, x_max, y_max]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # compute intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # compute areas
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # compute union
    union = area_box1 + area_box2 - intersection
    
    return intersection / union if union > 0 else 0


def match_predictions_to_ground_truth(
    predictions: list[dict],
    ground_truths: list[dict],
    img_height: float,
    img_width: float,
    iou_threshold: float = 0.5
) -> set[list[dict]]:
    """Categorizes predicted bboxes into TP, FP & FN based on IoU with bbox in ground truth"""
    
    tp = []  # True Positives
    fp = []  # False Positives
    fn = []  # False Negatives
    used_ground_truths = set()

    for pred in predictions:
        best_iou = 0
        best_gt = None
        
        pred_bbox = pred["bbox"]
        pred_class = pred["class_id"]
        
        for i, gt in enumerate(ground_truths):
            if i in used_ground_truths:
                continue  # skip already matched ground truth
            
            gt_bbox = gt["bbox"]
            gt_class = gt["class_id"]
            
            if pred_class != gt_class:
                continue  # skip if classes are not the same
            
            pred_converted = convert_bbox(pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3], img_width, img_height)
            gt_converted = convert_bbox(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], img_width, img_height)
            
            iou = compute_iou(pred_converted, gt_converted)
            
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        
        if best_iou >= iou_threshold and best_gt is not None:
            tp.append((pred, ground_truths[best_gt], best_iou))
            used_ground_truths.add(best_gt)
        else:
            fp.append((pred, None, 0))

    # any unmatched gt are False Negatives
    for i, gt in enumerate(ground_truths):
        if i not in used_ground_truths:
            fn.append((None, gt, 0))

    return tp, fp, fn


def calculate_metrics(
    tp: int, 
    fp: int, 
    fn: int
) -> dict[str, float]:
    """Computes Precision, Recall, Accuracy and F1 based on TP, FP & FN"""
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    if tp + fp + fn > 0:
        accuracy = tp / (tp + fp + fn)
    else:
        accuracy = 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def draw_bounding_box(
    image: Image, 
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    class_id: float,
    similarity: float,
    color: set[float]
) -> Image:
    """Draws bbox on Image and labels class_id & Similarity score"""
    
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
    
    label = f"{class_id}: {similarity:.2f}"
    
    text_bbox = draw.textbbox((0, 0), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = x_min
    text_y = y_min - text_height if y_min - text_height > 0 else y_min
    
    draw.rectangle(
        [text_x, text_y, text_x + text_width, text_y + text_height], 
        fill=color
    )
    draw.text(
        (text_x, text_y), label, fill="white"
    )
    
    return image


def plot_similarity_histogram(
    similarities: list[float],
    save_path: str,
    sim_min: float = -1.0,
    sim_max: float = 1.0,
    bin_size: float = 0.1,
    log_scale: bool = False
) -> bool:
    """Plots a Histogram of similarities and saves to save_path"""
    
    try:
        bin_edges = np.arange(sim_min, sim_max + bin_size, bin_size)
        hist, edges = np.histogram(similarities, bins=bin_edges)

        plt.figure(figsize=(10, 6))

        for i in range(len(hist)):
            color = 'red' if edges[i + 1] <= 0 else 'green'
            plt.bar(edges[i], hist[i], width=bin_size, color=color, edgecolor=color, align='edge', alpha=0.7)

        plt.title(f"Histogram of Semantic Similarity Categorized into Bins of Size {bin_size}", fontsize=14)
        plt.xlabel("Semantic Similarity", fontsize=12)
        plt.ylabel("Frequency (Log Scale)" if log_scale else "Frequency", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(bin_edges, rotation=45)

        if log_scale:
            plt.yscale('log')

        plt.tight_layout()
        plt.savefig(save_path)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
