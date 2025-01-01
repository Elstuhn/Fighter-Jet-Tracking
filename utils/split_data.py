import os
import json
import shutil
import random
from tqdm import tqdm

def split_coco_dataset(coco_json_path, images_folder, output_folder, train_ratio=0.8):
    """
    Splits a COCO JSON dataset into training and validation sets.

    :param coco_json_path: Path to the COCO JSON file.
    :param images_folder: Path to the folder containing images.
    :param output_folder: Path to the output folder for split datasets and images.
    :param train_ratio: Ratio of images to include in the training set.
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Helper function to copy images
    def copy_images(image_list, destination_folder):
        for image in tqdm(image_list, desc=f"Copying images to {destination_folder}"):
            src_path = os.path.join(images_folder, image["file_name"])
            dest_path = os.path.join(destination_folder, image["file_name"])
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)

    # Copy images to respective folders
    copy_images(train_images, train_folder)
    copy_images(val_images, val_folder)

    # Generate new JSONs for train and val
    def create_split_json(image_list, annotations, categories):
        image_ids = {img["id"] for img in image_list}
        filtered_annotations = [ann for ann in annotations if ann["image_id"] in image_ids]
        return {
            "images": image_list,
            "annotations": filtered_annotations,
            "categories": categories
        }

    train_json = create_split_json(train_images, coco_data["annotations"], coco_data["categories"])
    val_json = create_split_json(val_images, coco_data["annotations"], coco_data["categories"])

    # Save JSON files
    with open(os.path.join(output_folder, "train.json"), 'w') as f:
        json.dump(train_json, f, indent=4)
    with open(os.path.join(output_folder, "val.json"), 'w') as f:
        json.dump(val_json, f, indent=4)

    print(f"Train and validation datasets created successfully in {output_folder}")

coco_json_path = "./coco_data.json"  
images_folder = "./Jet_Data/dataset" 
output_folder = "./dataset"  
split_coco_dataset(coco_json_path, images_folder, output_folder)
