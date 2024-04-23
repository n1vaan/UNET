import os
import json
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

image_directory = '/Users/nivaankaushal/Desktop/archive/data/batch_1'

annotation_file = '/Users/nivaankaushal/Desktop/archive/data/annotations.json'

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

def create_mask(width, height, polygons):
    mask = Image.new('L', (width, height), 0) 
    draw = ImageDraw.Draw(mask)

    for polygon in polygons:
        draw.polygon(polygon, outline=1, fill=1) 

    return mask

json_image_names = [img['file_name'] for img in annotations['images']]


for image_name in os.listdir(image_directory):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):  
        base_name = os.path.splitext(image_name)[0]

        image_info = next((img for img in annotations['images'] if img['file_name'] == "batch_1/"+image_name), None)
        if image_info:
            print("Found!")
            image_id = image_info['id']

            annotation_info = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

            polygons = [ann['segmentation'][0] for ann in annotation_info if 'segmentation' in ann]

            width, height = image_info['width'], image_info['height']

            mask = create_mask(width, height, polygons)
            image_path = image_directory+"/"+image_name
            original_image = Image.open(image_path)

            plt.imshow(mask, cmap="gray")
            plt.pause(1)  

            output_directory = '/Users/nivaankaushal/Desktop/masks/batch_1'
            os.makedirs(output_directory, exist_ok=True)  

            mask_path = os.path.join(output_directory, f'mask_{base_name}.jpg')

            mask.save(mask_path)

            print(f"Mask saved for image {image_name} at {mask_path}")
        else:
            print(f"Could not find annotations for image {image_name}")
    else:
        print("Failed!")