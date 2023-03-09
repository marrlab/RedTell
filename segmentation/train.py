import glob
import torch
import os
from PIL import Image
from . import datasets
from . import segmentation_utils  as su
from . import transforms as T
from . import utils as ut
from . import engine
import numpy as np

def get_train_test_datasets(images_dir, fraction_train_images=0.8):

  # create dataloaders
  images_paths = glob.glob(os.path.join(images_dir, "images", "*.tif"))

  train_images, test_images = su.train_validation_split(images_paths, 
                            fraction_train_images=fraction_train_images)
                      

  train_dataset = datasets.CellSegmentationDataset(train_images, su.get_transform(train=True))
  data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=2,
        collate_fn=ut.collate_fn)
  
  if fraction_train_images < 1:
    test_dataset = datasets.CellSegmentationDataset(test_images, su.get_transform(train=False))
    data_loader_val = torch.utils.data.DataLoader(
          test_dataset, batch_size=1, shuffle=False, num_workers=2,
          collate_fn=ut.collate_fn)
  else: 
    data_loader_val = None
  
  return data_loader_train, data_loader_val


def perform_training(data_loader_train, num_classes=2, num_epochs=15):

    # training setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = su.get_instance_segmentation_model(num_classes)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=5,
                                                    gamma=0.05)

    # train model
    print("Training model")
    for epoch in range(num_epochs):
      engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
      lr_scheduler.step()

    return model

def evaluate_model(model, data_loader_test):

    print("Evaluating model on the test set")
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    engine.evaluate(model, data_loader_test, device=device)

def visualize_and_save_model_predictions(model, data_loader_test):

    print("Visualizing and saving predictions in data/test_segmentation")
    if not os.path.exists('./data/test_segmentation'):
          os.makedirs('./data/test_segmentation')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for idx, batch in enumerate(data_loader_test):
      image, _ =  batch
      image = image[0]

      with torch.no_grad():
        prediction = model([image.to(device)])

      masks = prediction[0]['masks'][:,0].cpu().numpy()

      prob_threshold = 0.5
      image_masks = np.zeros((572,572))

      for i in range(0, masks.shape[0]):
        mask_i = masks[i, :, :]
        mask_i = np.where(mask_i >= prob_threshold, i+1, 0)
        image_masks += mask_i

      image_masks = image_masks.astype(np.uint8)

      image = image.mul(255).permute(1, 2, 0).byte().numpy()
      image = np.array(Image.fromarray(np.squeeze(image, axis=2)).convert('RGB'))

      su.visualize_predictions(image, image_masks, 
                            "./data/test_segmentation/test_image" + str(idx) + ".tif")

def train_new_model(images_dir, model_name, num_classes=2, num_epochs=15):

  print("Loading data from " + images_dir + " for training")

  # 80,20 split
  print("Using 80% of data for training and 20% for testing to estimate model generalization ability")
  train_dataloader, test_dataloader =  get_train_test_datasets(images_dir)
  model = perform_training(train_dataloader, num_classes, num_epochs)
  evaluate_model(model, test_dataloader)
  visualize_and_save_model_predictions(model, test_dataloader)

  # training on all data
  print("Training model on the whole dataset")
  train_dataloader, _ =  get_train_test_datasets(images_dir, fraction_train_images=1.0)
  model = perform_training(train_dataloader, num_classes, num_epochs)
  # save model
  print("Saving model as " +  model_name + ".model in segmentation/models")
  torch.save(model.state_dict(), "segmentation/models/" + model_name + ".model")


