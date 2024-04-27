import os
import json
from PIL import Image, ImageDraw, ImageOps

image_directory = 'add_path'
def read_and_copy(image_directory: str, batch:str):

    annotation_file = 'annotations.json'

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    def create_mask(width, height, polygons):
        mask = Image.new('L', (width, height), 0) 
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            draw.polygon(polygon, outline=1, fill=255) 

        return mask

    json_image_names = [img['file_name'] for img in annotations['images']]


    for image_name in os.listdir(image_directory):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):  
            base_name = os.path.splitext(image_name)[0]

            image_info = next((img for img in annotations['images'] if img['file_name'] == batch + "/"+image_name), None)
            if image_info:
                print("Found!")
                image_id = image_info['id']

                annotation_info = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

                polygons = [ann['segmentation'][0] for ann in annotation_info if 'segmentation' in ann]

                width, height = image_info['width'], image_info['height']

                mask = create_mask(width, height, polygons)
                image_path = image_directory+"/"+image_name
                original_image = Image.open(image_path).convert("RGBA")
                mask_rgba = ImageOps.colorize(mask, (0, 0, 0), (0, 255, 0))  # Red mask



                # Convert mask to RGBA with specific color
                mask_rgba = ImageOps.colorize(mask, (0, 0, 0), (9, 237, 70))  
                mask_rgba = mask_rgba.convert("RGBA")  # Ensure 'RGBA' mode

                # Load the original image
                image_path = os.path.join(image_directory, image_name)
                original_image = Image.open(image_path).convert("RGBA")  # Ensure 'RGBA' mode

                # Ensure the mask and original image have the same size
                if original_image.size != mask_rgba.size:
                    mask_rgba = mask_rgba.resize(original_image.size)

                # Use Image.composite to overlay the mask on the image
                overlaid_image = Image.composite(mask_rgba, original_image, mask).convert("RGB")

                output_directory = '/masks/' + batch
                os.makedirs(output_directory, exist_ok=True)  

                mask_path = os.path.join(output_directory, f'mask_{base_name}.jpg')
                overlaid_image.save(mask_path)


                print(f"Mask saved for image {image_name} at {mask_path}")
            else:
                print(f"Could not find annotations for image {image_name}")
        else:
            print("Failed!")

lst = []
for x in range(15):
    lst.append("batch_" + str(x+1))

print(lst)

for item in lst:
    read_and_copy(image_directory+item, item)
