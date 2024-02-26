import os
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.feature import peak_local_max 
from scipy.spatial import distance_matrix
from tqdm import tqdm
import glob


def find_local_max(img, 
                  median_filter_size=3, 
                  min_distance_between_local_minima=10):
    
    vesicle_img = ndimage.median_filter(img, size=median_filter_size)
    coordinates = peak_local_max(vesicle_img, min_distance=min_distance_between_local_minima)

    return coordinates

def find_vesicles(ca_img, mask, vesicle_radius=4, neighborhood_radius=8, int_thres=1):
    
    max_coords =  find_local_max(ca_img)
         
    vesicles = []

    for i in range(max_coords.shape[0]):

      if mask[max_coords[i][0], max_coords[i][1]] > 0:

        vesicle_circle = ca_img[max_coords[i][0] - vesicle_radius : max_coords[i][0] + vesicle_radius,
                                   max_coords[i][1] - vesicle_radius : max_coords[i][1] + vesicle_radius]                         
        mean_vesicle = np.mean(vesicle_circle)
    
        neighborhood_circle = ca_img[max_coords[i][0] - neighborhood_radius : max_coords[i][0] + neighborhood_radius,
                                   max_coords[i][1] - neighborhood_radius : max_coords[i][1] + neighborhood_radius]

        mean_neigh = (np.sum(neighborhood_circle) - np.sum(vesicle_circle))/((2 * neighborhood_radius) ** 2 - (2 * vesicle_radius) ** 2)

        if mean_vesicle - mean_neigh > int_thres:
          vesicles.append([max_coords[i][0], max_coords[i][1]])
    
    return np.array(vesicles)


def count_vesicles(mask, vesicles):

    num_vesicles = 0
    for j in range(vesicles.shape[0]):
      if mask[vesicles[j][0], vesicles[j][1]] > 0:
          num_vesicles+=1

    return num_vesicles

def visualize_vesicles(bf_img, ca_img, vesicles, save_path):

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(bf_img, cmap="gray")
    ax.imshow(ca_img, alpha=0.8)
    for vesicle in vesicles: 

      rect = plt.Rectangle(
              (vesicle[1]-10, vesicle[0]-10),
              20,
              20,
              fill=False,
              color = "white",
              linewidth=1.5)
      ax.add_patch(rect)

    ax.axis('off')
        
    fig.tight_layout()
    # plt.show()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    

def extract_vesicle_number(img_dir):

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif*")))
    mask_paths = sorted(glob.glob(os.path.join(os.path.dirname(img_dir), "masks", "*.tif*")))

    vesicle_features = []

    # create directory for vesicle detection results
    if not os.path.exists(os.path.join(os.path.dirname(img_dir), 'vesicle_results')):
        os.makedirs(os.path.join(os.path.dirname(img_dir), 'vesicle_results')) 

    for img_path, mask_path in tqdm(zip(img_paths, mask_paths)):

      print(img_path, mask_path)

      assert img_path.split("/")[-1] == mask_path.split("/")[-1]
      ca_img = np.asarray(Image.open(img_path).convert('L'))
      mask = np.array(Image.open(mask_path))

      vesicles = find_vesicles(ca_img, mask)

      # for every mask count vesicles
      num_masks = np.max(mask)

      for i in range(1, num_masks+1):
        mask_i = np.where(mask==i, mask, 0)
        num_vesicles = count_vesicles(mask_i, vesicles)
        
        img_name = os.path.join(os.path.dirname(img_dir), "images", img_path.split("/")[-1])

        vesicle_features.append({"image": img_name,
        "cell_id": i, 
        "fluo-4_num_visicles":num_vesicles})

      # save vesicle results
      img_name = mask_path.split("/")[-1]
      bf_img_path = os.path.join(os.path.dirname(img_dir), "images", img_name)
      bf_img = np.asarray(Image.open(bf_img_path).convert('L'))
      save_results_path = os.path.join(os.path.dirname(img_dir), "vesicle_results", img_name)

      visualize_vesicles(bf_img, ca_img, vesicles, save_results_path)
      
    print("Images with detected vesicles are saved in " + os.path.join(os.path.dirname(img_dir), "vesicle_results"))
    
    return vesicle_features



