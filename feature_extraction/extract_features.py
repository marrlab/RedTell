import logging
import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
from PIL import Image
from radiomics import featureextractor
from skimage.measure import regionprops_table
from functools import reduce
from . import select_features  as sf
from . import detect_vesicles as dv
 
logging.getLogger('radiomics').setLevel(logging.CRITICAL+2)

#logger = logging.getLogger('my-logger')
#logger.propagate = False

def initialize_feature_extractor():
  # Instantiate the extractor
  settings = {'force2D': True,'normalize':False }
  extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
  extractor.enableFeatureClassByName('shape2D')

  return extractor

def initialize_region_props():
  properties = ('bbox_area', 'convex_area', 'extent',
                  'eccentricity', 'equivalent_diameter', 'solidity')
  return properties

def extract_features_for_cells_in_single_img(img_path, mask_path, 
                                              feature_extractor, region_props):

  img = np.asarray(Image.open(img_path).convert('L'))
  sitk_img = sitk.GetImageFromArray(img, 0)
  masks = np.array(Image.open(mask_path))

  num_masks = np.max(masks)

  extracted_features = []

  for i in range(1, num_masks+1):
 
    mask = np.where(masks == i, 1, 0)

    if np.sum(mask) > 10:

      cell_pos = np.where(mask)
      cell_bbox = [np.min(cell_pos[1]), np.min(cell_pos[0]),
                        np.max(cell_pos[1]), np.max(cell_pos[0])]
      cell_area = (cell_bbox[3] - cell_bbox[1]) * (cell_bbox[2] - cell_bbox[0])

      if cell_area > 0:

        # pyradiomics
        sitk_mask = sitk.GetImageFromArray(mask, 0)
        pyradiomics_results_raw = feature_extractor.execute(sitk_img, sitk_mask)
        pyradiomics_results = {}
        for feature in pyradiomics_results_raw.keys():
          try:
            pyradiomics_results[feature] = pyradiomics_results_raw[feature].item()
          except Exception as e:
            pyradiomics_results[feature] = pyradiomics_results_raw[feature]

        # regionprops
        regionprops_results = regionprops_table(mask, img, region_props)
        regionprops_results = {k: v[0] for k, v in regionprops_results.items()}

        # merge properties
        cell_features = {**pyradiomics_results, **regionprops_results}

        # add cell and image information:
        img_dir = os.path.dirname(os.path.dirname(img_path))
        img_name = os.path.join(img_dir, "images", img_path.split("/")[-1])
        cell_features['image'] = img_name
        cell_features['cell_id'] = i

        extracted_features.append(cell_features)

  return extracted_features

def extract_features_for_single_channel(img_dir):

  feature_extractor = initialize_feature_extractor()
  region_props = initialize_region_props()

  img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
  mask_paths = sorted(glob.glob(os.path.join(os.path.dirname(img_dir), "masks", "*.tif")))


  dataset_features = []

  for img_path, mask_path in tqdm(zip(img_paths, mask_paths)):

    assert img_path.split("/")[-1] == mask_path.split("/")[-1]

    img_features = extract_features_for_cells_in_single_img(img_path, mask_path,
                                                            feature_extractor, region_props)
    dataset_features.extend(img_features)

  return dataset_features

def create_feature_table(extracted_features, feature_dict):

  df_extracted_features = pd.DataFrame(extracted_features)
  keep_features = ["image", "cell_id"] + list(feature_dict.keys())
  df_extracted_features = df_extracted_features[keep_features]
  df_extracted_features = df_extracted_features.rename(columns=feature_dict)
  return df_extracted_features

def extract_features(img_dir, channel_list):

    feature_tables = []
    feature_dict = sf.get_feature_dict("feature_extraction/feature_list.json")

    num_images = len(os.listdir(os.path.join(img_dir, "masks")))

    if "mask" in channel_list:
      print("Extracting shape features from", num_images ,"images")
      mask_features = extract_features_for_single_channel(os.path.join(img_dir, "images"))
      mask_dict = sf.get_features_to_extract(feature_dict, "mask")
      mask_features_table = create_feature_table(mask_features, mask_dict)

      feature_tables.append(mask_features_table)
      channel_list.remove("mask")

    if "bf" in channel_list:
      print("Extracting features from brightfield channel from", num_images ,"images")
      bf_features = extract_features_for_single_channel(os.path.join(img_dir, "images"))
      bf_dict = sf.get_features_to_extract(feature_dict, "bf")
      bf_features_table = create_feature_table(bf_features, bf_dict)
      bf_features_table = bf_features_table.add_prefix('bf_')
      bf_features_table = bf_features_table.rename(columns={"bf_image":"image",
                                                            "bf_cell_id": "cell_id"})
      feature_tables.append(bf_features_table)
      channel_list.remove("bf")

    if "fluo-4" in channel_list:
      print("Extracting features from Fluo-4 channel from", num_images ,"images")
      ca_features = extract_features_for_single_channel(os.path.join(img_dir, "fluo-4"))
      ca_dict = sf.get_features_to_extract(feature_dict,"fl")
      ca_features_table = create_feature_table(ca_features, ca_dict)

      # append vesicles
      print("Couning vesicles from Fluo-4 channel from", num_images ,"images")
      vesicle_features = dv.extract_vesicle_number(os.path.join(img_dir, "fluo-4"))
      vesicle_features_table = pd.DataFrame(vesicle_features)

      ca_features_table = pd.merge(vesicle_features_table,ca_features_table,on=["image", "cell_id"],
                                              how="inner")
      ca_features_table = ca_features_table.add_prefix('fluo-4_')
      ca_features_table = ca_features_table.rename(columns={"fluo-4_image":"image",
                                                            "fluo-4_cell_id": "cell_id"})

      feature_tables.append(ca_features_table)
      channel_list.remove("fluo-4")

    # other channels
    if len(channel_list) > 0:
      for channel in channel_list:
        # check if data available
        if os.path.exists(os.path.join(img_dir, channel)):

          print("Extracting features from " + channel + " channel from", num_images ,"images")
          ch_features = extract_features_for_single_channel(os.path.join(img_dir, channel))
          ch_dict = sf.get_features_to_extract(feature_dict,"fl")
          ch_features_table = create_feature_table(ch_features, ch_dict)
          ch_features_table = ch_features_table.add_prefix(channel+"_")
          ch_features_table = ch_features_table.rename(columns={channel + "_image":"image",
                                                            channel + "_cell_id": "cell_id"})
          feature_tables.append(ch_features_table)

        else:
          print("ERROR: no data available for channel", channel)

    # append features in feature table
    if len(feature_tables)>1:
      all_features_table = reduce(lambda left,right: pd.merge(left,right,on=["image", "cell_id"],
                                              how="inner"), feature_tables)
    elif len(feature_tables) == 1:
      all_features_table = feature_tables[0]

    else: 
      print("ERROR: no features could be extracted")
      return None

    table_name = os.path.join(img_dir, "features.csv")
    print("Finished")
    print("Extracted features are saved as " + table_name)
    all_features_table.to_csv(table_name, index=False)

    
  


    
  


