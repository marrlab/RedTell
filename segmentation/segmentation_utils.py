from . import utils
from . import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
import colorsys
import random
import matplotlib.pyplot as plt
import numpy as np


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_instance_segmentation_model(num_classes=2):
    # TODO: make sure we have the proper number of detected cells!!!!
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
                                                               box_detections_per_img=200,
                                                               box_nms_thresh = 0.25,
                                                               box_score_thresh=0.05)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # TODO: remove warningns

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def train_validation_split(images_paths, fraction_train_images=1.0):

  random.shuffle(images_paths)

  num_images = len(images_paths)
  num_train_images = int(len(images_paths) * fraction_train_images +1)
  train_images = images_paths[:num_train_images]
  validation_images = images_paths[num_train_images:]

  return train_images, validation_images

# from stardist
def generate_colors(num_colors):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 0.7
  hsv = [(i / num_colors, 1, brightness) for i in range(num_colors)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  perm = list(range(0,num_colors))
  random.shuffle(perm)
  colors = [colors[idx] for idx in perm]
  return colors


def bbox(mask):
    a = np.where(mask != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

# from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] * (1 - alpha) + alpha * color[c]*255,
                              image[:, :, c])
  return image


def visualize_predictions(img, masks, save_path):

  fig, ax = plt.subplots(1, dpi=120, figsize=(10,10))

  num_masks = np.max(masks)
  colors = generate_colors(num_masks)
  
  # annotate
  for i in range(num_masks+1):

      mask = np.where(masks==i+1, 1, 0)
      if len(np.unique(mask))>1:

        # add mask
        img = apply_mask(img, mask, colors[i], alpha=0.3)

        # add label
        x,y,h,w = bbox(mask)

        ax.annotate(i+1, ((h+w)/2, (x+y)/2), color='black', weight='bold',
                        fontsize=12, ha='center', va='center', alpha=1.0)
  ax.imshow(img)
  ax.set_axis_off()
  plt.axis('off')
  fig.tight_layout()
  # save
  fig.savefig(save_path, dpi=120)
  plt.close(fig)