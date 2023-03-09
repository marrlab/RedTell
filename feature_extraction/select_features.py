import json


# 'feature_list.json'
def get_feature_dict(feature_file):
  with open(feature_file) as json_file:
    feature_group_dict = json.load(json_file)
  return feature_group_dict

def get_features_to_extract(feature_group_dict, channel):

  if channel == "mask":
    feature_dict = {**feature_group_dict["pyradiomics_shape"],
                  **feature_group_dict["regioprops_shape"],
                }

  elif channel == "bf":
    feature_dict = {**feature_group_dict["intensity"],
                  **feature_group_dict["glcm"],
                  **feature_group_dict["gldm"],
                  **feature_group_dict["glrlm"],
                  **feature_group_dict["glszm"],
                  }

  elif channel == "fl":
    feature_dict = {**feature_group_dict["intensity"],
                **feature_group_dict["glcm"],
                **feature_group_dict["gldm"],
                **feature_group_dict["glrlm"],
                **feature_group_dict["glszm"],
                }

  elif channel == "fluo-4":
    feature_dict = {
                **feature_group_dict["intensity"],
                **feature_group_dict["glcm"],
                **feature_group_dict["gldm"],
                **feature_group_dict["glrlm"],
                **feature_group_dict["glszm"],
                **feature_group_dict["fluo-4"],
                }
  else:
    print("Error: Channel name is not valid")
    feature_dict = None

  return feature_dict

