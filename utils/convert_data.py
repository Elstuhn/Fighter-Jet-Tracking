import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm

def csv_to_coco(folder_path, output_json):
    """
    Converts a folder of CSV and image files to COCO dataset format.

    :param folder_path: Path to the folder containing images and CSV files.
    :param output_json: Path to the output COCO JSON file.
    """
    # COCO dataset structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Mapping for categories
    category_mapping = {}
    annotation_id = 0

    # Iterate through all files in the folder
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(folder_path, file)
            image_file = file.replace(".csv", ".jpg")
            image_file_path = os.path.join(folder_path, image_file)

            # Skip if the corresponding image doesn't exist
            if not os.path.exists(image_file_path):
                print(f"Image file {image_file} not found, skipping...")
                continue

            # Read image dimensions
            with Image.open(image_file_path) as img:
                width, height = img.size

            # Add image information to COCO
            image_id = len(coco["images"]) + 1
            coco["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })

            # Read CSV file for annotations
            df = pd.read_csv(csv_file_path)

            for _, row in df.iterrows():
                category_name = row['class']
                
                # Add category if not already added
                if category_name not in category_mapping:
                    category_id = len(category_mapping) + 1
                    category_mapping[category_name] = category_id
                    coco["categories"].append({
                        "id": category_id,
                        "name": category_name,
                        "supercategory": "none"
                    })
                else:
                    category_id = category_mapping[category_name]

                # Add annotation
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save COCO JSON to file
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=4)

    print(f"COCO dataset saved to {output_json}")

# Example usage
folder_path = "./Jet_Data/dataset"  # Replace with the path to your folder
output_json = "./coco_data.json"  # Replace with the desired output JSON path
csv_to_coco(folder_path, output_json)
