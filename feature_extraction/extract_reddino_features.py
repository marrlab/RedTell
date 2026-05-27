import glob
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import timm
import pandas as pd

# removes the masks that are empty
def bbox(mask):
    a = np.where(mask != 0)

    if len(a[0]) == 0:
        return None

    x_min = np.min(a[0])
    x_max = np.max(a[0])
    y_min = np.min(a[1])
    y_max = np.max(a[1])

    return x_min, x_max, y_min, y_max

def load_model():
    model = timm.create_model("hf_hub:Snarcy/RedDino-base", pretrained=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

def preprocess_image_tensors():
    return transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def extract_reddino_features(img_dir):

    combined_cell_data = [] # List to store dictionaries for DataFrame rows
    current_unique_cell_idx = 0

    img_paths = sorted(glob.glob(os.path.join(img_dir, "images", "*.tif")))
    mask_paths = sorted(glob.glob(os.path.join(img_dir, "masks", "*.tif")))

    model, device = load_model()
    transform = preprocess_image_tensors()

    for img_path_orig, mask_path_orig in zip(img_paths, mask_paths): # Renamed for clarity

        image = np.array(Image.open(img_path_orig))
        mask = np.array(Image.open(mask_path_orig))

        labels = np.unique(mask)
        labels = labels[labels != 0]  # remove background

        for label in labels:

            cell_mask = (mask == label)

            # Get bounding box coordinates for the cell
            coords = bbox(cell_mask)

            if coords is None:
                continue

            x_min, x_max, y_min, y_max = coords

            # Extract the image crop
            crop_array = image[x_min:x_max, y_min:y_max]

            # ignore crops that are too small
            if crop_array.shape[0] < 20 or crop_array.shape[1] < 20:
                continue

            # Convert the crop to a PIL Image and preprocess for the model
            pil_crop_image = Image.fromarray(crop_array).convert("RGB")

            # Preprocess the image and convert to tensor
            input_tensor = transform(pil_crop_image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding_tensor = model(input_tensor)

            embedding_features = embedding_tensor.cpu().numpy().flatten().tolist()

            row_data = {
                'cell_id': current_unique_cell_idx,
                'image_crop': pil_crop_image, # Storing PIL Image object
                'original_image_path': img_path_orig, # New metadata
                'original_mask_path': mask_path_orig, # New metadata
                'original_mask_label': label, # New metadata
                'bbox_x_min': x_min, # New metadata
                'bbox_x_max': x_max, # New metadata
                'bbox_y_min': y_min, # New metadata
                'bbox_y_max': y_max, # New metadata
            }
            # Add embedding features dynamically
            for i, feature_val in enumerate(embedding_features):
                row_data[f'feature_{i}'] = feature_val

            combined_cell_data.append(row_data)
            current_unique_cell_idx += 1

    # Create the DataFrame that holds all information
    features_table = pd.DataFrame(combined_cell_data)

    table_name = os.path.join(img_dir, "features_reddino.csv")
    print("Finished")
    print("Extracted features are saved as " + table_name)
    features_table.to_csv(table_name, index=False)
