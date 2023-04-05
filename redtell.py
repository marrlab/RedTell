import glob
import argparse
from segmentation.predict import segment_images
from segmentation.train import train_new_model
from feature_extraction.extract_features import extract_features
from annotation.generate_annotations import create_annotation_cells


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--funct', type=str, required=True)
  parser.add_argument('--data', type=str, required=True)
  parser.add_argument('--model', type=str, required=False)
  parser.add_argument('--channel', type=str, required=False, nargs="*")
  parser.add_argument('--num_cells', type=int, required=False)

  args = parser.parse_args()

  funct = args.funct
  data = args.data
  model = args.model
  channels = args.channel
  num_cells = args.num_cells


  if funct == "segment":
    if model == None:
      model = "mask_rcnn_commitment"
    segment_images(data, model)

  elif funct == "train_segmentation":
      train_new_model(data, model, num_epochs=15)

  elif funct == "extract_features":
    if channels == None:
      channels = ["mask", "bf"]
    extract_features(data, channels)

  elif funct == "annotate":
    if num_cells == None:
      num_cells = 200
    create_annotation_cells(data, num_cells)


  else:
    print("No such function.")

