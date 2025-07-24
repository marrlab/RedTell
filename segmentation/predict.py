import os
import torch
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from . import segmentation_utils as su
from scipy.ndimage import zoom
plt.ioff()



def segment_patch(patch, model, device, prob_threshold, mask_start_num=1,
                  min_cell_area = 30, border_threshold = 3):

  torch_img = patch/255
  torch_img = torch.as_tensor(torch_img, dtype=torch.float32)
  torch_img = torch_img.unsqueeze(0)
   
  with torch.no_grad():

    prediction = model([torch_img.to(device)])
    masks = prediction[0]['masks'][:,0].cpu().numpy()
    
    image_masks = np.zeros(patch.shape)

    for i in range(0, masks.shape[0]):

      mask_i = masks[i, :, :]
      
      # Quality control: keep cell mask only if it is big enouth
      if np.sum(np.where(mask_i >= prob_threshold, 1, 0)) > min_cell_area:
        mask_i = np.where(mask_i >= prob_threshold, i + mask_start_num, 0)

        # remove of too close to the patch borders
        # top
        if np.max(mask_i[:border_threshold, :]) > 0:
          continue
        # bottom
        if np.max(mask_i[-border_threshold:, :]) > 0:
          continue
        # left
        if np.max(mask_i[:, :border_threshold]) > 0:
          continue
        # right
        if np.max(mask_i[:, -border_threshold:]) > 0:
          continue    

        # Add only where not already occupied
        image_masks += mask_i
        image_masks = np.where(image_masks > i + mask_start_num, i + mask_start_num, image_masks)

  image_masks = image_masks.astype(np.uint16)
  return image_masks


def segment_images(img_dir, model):

  print("Segmenting images in ", img_dir)
  print("Using model ", model)

  num_classes = 2
  prob_threshold = 0.3
  model_path = os.path.join("segmentation/models", model+".model")
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # define directory to store the results
  if not os.path.exists(os.path.join(img_dir, 'segmentation_results')):
        os.makedirs(os.path.join(img_dir, 'segmentation_results'))

  if not os.path.exists(os.path.join(img_dir, 'masks')):
        os.makedirs(os.path.join(img_dir, 'masks'))

  # create model
  model = su.get_instance_segmentation_model(num_classes)

  if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
  else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
  model.to(device)
  model.eval()

      # define directory to store the results
  if not os.path.exists(os.path.join(img_dir, 'segmentation_results')):
        os.makedirs(os.path.join(img_dir, 'segmentation_results'))

  if not os.path.exists(os.path.join(img_dir, 'masks')):
        os.makedirs(os.path.join(img_dir, 'masks'))

  # define images
  image_paths = glob.glob(os.path.join(img_dir, "images", "*.tif"))

  print("Segmenting ",len(image_paths) , " images")


  for img_path in tqdm(image_paths):
    
    img_raw = Image.open(img_path)
    img_gray = img_raw.convert("L")
    img = np.array(img_gray)

    size = img_gray.size

    patch_heigth = int(size[0]/2)
    patch_width = int(size[1]/2)

    patch_top_left = img[:patch_width, :patch_heigth]
    masks_top_left = segment_patch(patch_top_left, model, device, prob_threshold)
    num_masks = np.max(masks_top_left) + 1 

    patch_bottom_left = img[patch_width:, :patch_heigth]
    masks_bottom_left = segment_patch(patch_bottom_left, model, device, 
                                      prob_threshold, num_masks)
    num_masks = np.max(masks_bottom_left) + 1 

    patch_top_right = img[:patch_width, patch_heigth:]
    masks_top_right = segment_patch(patch_top_right, model, device, 
                                      prob_threshold, num_masks)
    num_masks = np.max(masks_top_right) + 1 

    patch_bottom_right = img[patch_width:, patch_heigth:]
    masks_bottom_right = segment_patch(patch_bottom_right, model, device, 
                                      prob_threshold, num_masks)

    # aggregate patches masks
    image_masks = np.zeros((patch_width*2, patch_heigth*2))

    print(patch_top_right.shape, masks_top_right.shape)
    print(patch_bottom_right.shape, masks_bottom_right.shape)
    print(patch_top_left.shape, masks_top_left.shape)
    print(patch_bottom_left.shape, masks_bottom_left.shape)

    # aggregate top left and top right
    for i in range(patch_width):
      if  (masks_top_left[i, patch_heigth-3] != 0) and (masks_top_right[i, 3]!= 0):
        mask_top_left_id = masks_top_left[i, patch_heigth-3]
        mask_top_right_id = masks_top_right[i, 3]
        masks_top_right = np.where(masks_top_right == mask_top_right_id, mask_top_left_id, masks_top_right)

    # aggregate bottom left and bottom right
    for i in range(patch_width):
      if  (masks_bottom_left[i, patch_heigth-3] != 0) and (masks_bottom_right[i, 3]!= 0):
        masks_bottom_left_id = masks_bottom_left[i, patch_heigth-3]
        masks_bottom_right_id = masks_bottom_right[i, 3]
        masks_bottom_right = np.where(masks_bottom_right == masks_bottom_right_id, masks_bottom_left_id, masks_bottom_right)

  # aggregate top left and bottom left
    for i in range(patch_heigth):
      if  (masks_top_left[patch_width-3, i] != 0) and (masks_bottom_left[3, i]!= 0):
        mask_top_left_id = masks_top_left[patch_width-3, i]
        masks_bottom_left_id = masks_bottom_left[3, i]
        masks_bottom_left = np.where(masks_bottom_left == masks_bottom_left_id, mask_top_left_id, masks_bottom_left)

  # aggregate top right and bottom right
    for i in range(patch_heigth):
      if  (masks_top_right[patch_width-3, i] != 0) and (masks_bottom_right[3, i]!= 0):
        mask_top_right_id = masks_top_right[patch_width-3, i]
        masks_bottom_right_id = masks_bottom_right[3, i]
        masks_bottom_right = np.where(masks_bottom_right == masks_bottom_right_id, mask_top_right_id, masks_bottom_right)
    
    image_masks[:patch_width,:patch_heigth] = masks_top_left
    image_masks[patch_width:,:patch_heigth] = masks_bottom_left
    image_masks[:patch_width,patch_heigth:] = masks_top_right
    image_masks[patch_width:,patch_heigth:] = masks_bottom_right

  # remove cell detections on the borders
    filtered_image_masks = np.zeros((patch_width*2, patch_heigth*2))       
    border_threshold = 3

    masks_ids = list(np.unique(image_masks))
    masks_ids.remove(0)
    for mask_id in masks_ids:
      mask_i = np.where(image_masks == mask_id, mask_id, 0)
    # top
      if np.max(mask_i[:border_threshold, :]) > 0:
        continue
    # bottom
      if np.max(mask_i[-border_threshold:, :]) > 0:
        continue
    # left
      if np.max(mask_i[:, :border_threshold]) > 0:
        continue
    # right
      if np.max(mask_i[:, -border_threshold:]) > 0:
        continue
      filtered_image_masks = filtered_image_masks + mask_i  

    # harmonize ids
    filtered_image_masks = np.where(filtered_image_masks > 0, filtered_image_masks + 1000, 0)
    masks_ids = list(np.unique(filtered_image_masks))
    masks_ids.remove(0)
    new_id = 1
    for i, mask_id in  enumerate(masks_ids):
      filtered_image_masks = np.where(filtered_image_masks == mask_id, i+1, filtered_image_masks)

    image_masks = filtered_image_masks.astype(np.uint16)
    
    image = img_gray.convert('RGB')
    image = np.array(image)
    
    img_name = img_path.split("/")[-1]

    # resize back
    #zoom_coef = (original_size[0]/size[0], original_size[1]/size[1])
    #image_masks = zoom(image_masks, zoom_coef, order=3) 
    #image_masks = image_masks.astype(np.uint16)

    # save masks
    save_masks = Image.fromarray(image_masks)
    save_mask_path = os.path.join(img_dir, "masks", img_name)
    save_masks.save(save_mask_path)

    # save segmentation results on images
    save_results_path = os.path.join(img_dir, "segmentation_results", img_name)
    su.visualize_predictions(image, image_masks, save_results_path)



  print("Segmentation is done")
  print("Segmentation masks are stored in ", os.path.join(img_dir, "masks"))
  print("Segmentation results are stored in ", os.path.join(img_dir, "segmentation_results"))

