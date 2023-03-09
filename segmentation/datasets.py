from PIL import Image
import numpy as np
import scipy
import torch
import glob
import random
import configparser
import csv
import os
import os.path as osp
import pickle
import torchvision


class CellSegmentationDataset(torch.utils.data.Dataset):
    """ Data class for the Multiple Object Tracking Dataset
    """

    def __init__(self, img_paths, transforms=None):
        self.transforms = transforms
        self._classes = ('background', 'cell')
        self._img_paths = img_paths
        self.anno_dir = self._img_paths[0].split("/")[-3]

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):

        img_path = self._img_paths[idx]
        # determine image name
        img_name = img_path.split('/')[-1]
        # find image masks
        img_masks_path = os.path.join(self.anno_dir, "annotations", img_name)
        img_masks = np.array(Image.open(img_masks_path).convert('L'))
        num_cells = np.max(img_masks)

        masks = []
        bounding_boxes = []
        areas = []

        for i in range(1, num_cells+1):

          cell_mask  = np.where(img_masks == i, 1, 0)

          if np.sum(cell_mask) > 0:

            cell_pos = np.where(cell_mask)
            cell_bbox = [np.min(cell_pos[1]), np.min(cell_pos[0]),
                        np.max(cell_pos[1]), np.max(cell_pos[0])]

            cell_area = (cell_bbox[3] - cell_bbox[1]) * (cell_bbox[2] - cell_bbox[0])

            masks.append(cell_mask)
            bounding_boxes.append(cell_bbox)
            areas.append(cell_area)


        # convert everything into a torch.Tensor
        return {'boxes': torch.as_tensor(np.array(bounding_boxes), dtype=torch.float32),
                'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),
                'labels': torch.ones((num_cells,), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.as_tensor(areas, dtype=torch.int64),
                'iscrowd': torch.zeros((num_cells,), dtype=torch.int64),
                'visibilities': torch.ones((num_cells,), dtype=torch.int64)}

    def __getitem__(self, idx):

        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert('L')

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)