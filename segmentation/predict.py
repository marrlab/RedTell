import os
import torch
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from . import segmentation_utils as su
plt.ioff()


def segment_images(img_dir, model):

  print("Segmenting images in ", img_dir)
  print("Using model ", model)

  num_classes = 2
  prob_threshold = 0.5
  model_path = os.path.join("segmentation/models", model+".model")
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
          img = img_raw.convert("L")
          img = img.resize((572,572))
          img = np.array(img)
          torch_img = img/255
          torch_img = torch.as_tensor(torch_img, dtype=torch.float32)
          torch_img = torch_img.unsqueeze(0)


          with torch.no_grad():

              prediction = model([torch_img.to(device)])
              masks = prediction[0]['masks'][:,0].cpu()
              image_masks = np.zeros((572,572))

              for i in range(0, masks.shape[0]):
                  mask_i = masks[i, :, :]

                  # quality check 
                  if np.sum(np.where(mask_i >= prob_threshold, 1, 0)) > 20:
                    mask_i = np.where(mask_i >= prob_threshold, i+1, 0)
                    image_masks += mask_i


              image_masks = image_masks.astype(np.uint8)
              image = np.array(img_raw.convert('RGB'))
              img_name = img_path.split("/")[-1]

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

