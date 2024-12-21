import os
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import argparse


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_and_mask(image_path, mask_path):
    # Load image
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Convert mask to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    return image, mask


def generate_random_points_in_mask(mask, num_points):
    # Get indices where mask == 1
    indices = np.argwhere(mask == 1)
    if len(indices) == 0:
        raise ValueError("Mask is empty, no foreground pixels found.")
    # Randomly select num_points indices
    selected_indices = indices[np.random.choice(indices.shape[0], num_points, replace=False)]
    # Coordinates in (x, y)
    point_coords = selected_indices[:, [1, 0]]  # Switch columns to get (x, y)
    point_coords = point_coords.astype(np.float32)
    return point_coords


def generate_random_box_in_mask(mask):
    # Get indices where mask == 1
    indices = np.argwhere(mask == 1)
    if len(indices) == 0:
        raise ValueError("Mask is empty, no foreground pixels found.")
    # Get bounding box of the mask
    y_coords, x_coords = indices[:, 0], indices[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    # Optionally, we can randomize the box a bit within the mask
    # For simplicity, let's just use the bounding box
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    return box


def compute_iou_and_dice(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0.0
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) != 0 else 0.0
    return iou, dice


def overlay_mask_on_image(image, mask, alpha=0.5):
    # image: H x W x 3
    # mask: H x W
    color = np.array([0, 255, 0])  # Green color for the mask
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    return overlay.astype(np.uint8)


def load_predictor(model_type, checkpoint, device):
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device)
    predictor = SamPredictor(model)
    return predictor


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the ground truth mask')
    parser.add_argument('--model_type', type=str, default='vit_b', help='Model type (e.g., vit_b, vit_l, vit_h)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--prompt_type', type=str, choices=['point', 'box'], default='point', help='Type of prompt')
    parser.add_argument('--num_points', type=int, default=1,
                        help='Number of points to generate if prompt_type is point')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    # Load image and mask
    image, gt_mask = load_image_and_mask(args.image_path, args.mask_path)

    # Generate prompts
    if args.prompt_type == 'point':
        point_coords = generate_random_points_in_mask(gt_mask, args.num_points)
        point_labels = np.ones(args.num_points, dtype=np.int32)
        box = None
    else:
        point_coords = None
        point_labels = None
        box = generate_random_box_in_mask(gt_mask)

    # Load model
    device = torch.device(args.device)
    predictor = load_predictor(args.model_type, args.checkpoint, device)

    # Set image
    predictor.set_image(image)

    # Run prediction
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=False,
    )
    # masks is a list of masks, but since multimask_output=False, it should be a single mask
    pred_mask = masks[0]

    # Compute IOU and Dice
    iou, dice = compute_iou_and_dice(pred_mask, gt_mask)

    print(f'IOU: {iou:.4f}, Dice: {dice:.4f}')

    # Overlay mask on image
    overlay_image = overlay_mask_on_image(image, pred_mask)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_image)
    plt.title(f'IOU: {iou:.4f}, Dice: {dice:.4f}')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
