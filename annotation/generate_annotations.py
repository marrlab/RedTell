import os 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def create_annotation_table(data_dir, num_cells):

  assert os.path.exists(os.path.join(data_dir, "features.csv")), "ERROR: File with extracted features not available"

  data = pd.read_csv(os.path.join(data_dir, "features.csv"))

  assert data.shape[0] > 1, "ERROR: empty feature table"
  
  sample_data = data.sample(num_cells)
  sample_data["annotation_id"] =  list(range(1, num_cells+1))
  sample_data["label"] = ""
  sample_data = sample_data[["annotation_id", "label",  "image", "cell_id"]]
  sample_data.to_csv(os.path.join(data_dir, "annotations.csv"), index=False)

  return sample_data


def create_annotation_cells(data_dir, num_cells):

  num_cells = max(num_cells, 200)
  annotation_df = create_annotation_table(data_dir, num_cells)
  annotation_df = annotation_df[[ "annotation_id", "label", "cell_id", "image"]]

  if not os.path.exists(os.path.join(data_dir, 'annotations')):
    os.makedirs(os.path.join(data_dir, 'annotations')) 

  annotation_df.to_csv(os.path.join(data_dir, "annotations.csv"), index=False)

  print("Selecting and saving " + str(num_cells) + " cells for annotation")  

  for row in tqdm(annotation_df.itertuples(index=False)):

    anno_id = row[0]
    label = row[1]
    img_path = row[3]
    cell_id= row[2]

    img = np.asarray(Image.open(img_path).convert('L'))
    mask_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), "masks", img_path.split("/")[-1])
    mask = np.array(Image.open(mask_path))

    # locate cell on the imagee
    cell_mask = np.where(mask == cell_id, 1, 0)
    cell_pos = np.where(cell_mask)
    cell_bbox = [np.min(cell_pos[1]), np.min(cell_pos[0]),
                          np.max(cell_pos[1]), np.max(cell_pos[0])]

    # extract cell patch from image
    img_h, img_w = img.shape
      
    cell_img = img[max(0, cell_bbox[1]-10): min(cell_bbox[3]+10, img_h),
                  max(0, cell_bbox[0]-10): min(cell_bbox[2]+10, img_w)]

    cell_img = Image.fromarray(cell_img)
    img_name = os.path.join(data_dir, "annotations", str(anno_id)+".tif")
    cell_img.save(img_name)
