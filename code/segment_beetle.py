import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse
import torch

from utils import load_dataset_images, read_image_paths, get_sam_model
from segment_anything import SamAutomaticMaskGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing individual beetle images, ex: /User/micheller/images")
    parser.add_argument("--result", required=True, help="Directory containing result of the segmentation. ex: /User/micheller/result")
    return parser.parse_args()


def is_mask_closer_to_white(image, mask):
    # Calculate the average color of the masked area
    masked_area = image[mask]
    average_color = np.mean(masked_area, axis=0)
    # Calculate the Euclidean distance to white and black
    dist_to_white = np.linalg.norm(average_color - np.array([255, 255, 255]))
    dist_to_black = np.linalg.norm(average_color - np.array([0, 0, 0]))
    # Check if the masked area is closer to white than black
    return dist_to_white < dist_to_black

def crop_to_bb(image):
    # Get coordinates of the bounding box
    coords = np.argwhere(np.any(image != 0, axis=-1))
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return image[x0:x1, y0:y1]

def save_anns(anns, image, path):
    combined_mask = np.zeros_like(image, dtype=np.uint8)

    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        res = np.asarray(image) * np.repeat(mask[:, :, None], 3, axis=2)

        # Skip background regions
        if is_mask_closer_to_white(image, mask):
            continue

        # Combine the current mask with the combined mask using np.maximum
        combined_mask = np.maximum(combined_mask, res)

    # Crop the combined mask to its bounding box
    combined_mask_cropped = crop_to_bb(combined_mask)

    # Convert to BGR for saving
    combined_mask_bgr = cv2.cvtColor(combined_mask_cropped, cv2.COLOR_RGB2BGR)

    # Save the combined mask
    cv2.imwrite(path, combined_mask_bgr)


def main():
    args = parse_args()
    # Leverage GPU if available
    use_cuda = torch.cuda.is_available()
    DEVICE   = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device: ", DEVICE)

    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    
    # read images in
    dataset_folder = args.images + '/*'
    output_folder = args.result
    image_filepaths = read_image_paths(dataset_folder)
    print(f'Number of images in dataset: {len(image_filepaths)}')


    # load SAM model
    segmentation_model = get_sam_model(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(
        model=segmentation_model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        crop_n_layers=2,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  
    )

    # create masks using SAM (segment-anything-model)
    print("Segmenting beetle images...")
    for fp in image_filepaths:
        
        image = cv2.imread(fp)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        
        #create the path to which the mask will be saved
        output_path = os.path.join(output_folder, fp.split('/')[-1]) 
        
        os.makedirs(output_folder, exist_ok=True)
        
        #save mask with cv2 to preserve pixel categories
        print(f"Mask path:{output_path}")
        save_anns(masks, image, output_path)

    return


if __name__ == "__main__":
    main()