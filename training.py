import os, glob
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import skimage.io 
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import imgaug
from imgaug import parameters as iap
#os.environ["CUDA_VISIBLE_DEVICES"]="2"




class CellsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cells"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + tree

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class CellsDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_cells(self, dataset_dir):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("cell", 1, "cell")
        self.dataset_dir = dataset_dir
        
        image_ids = list(set(next(os.walk(dataset_dir))[1]))

        for _id in image_ids:
            self.image_path = os.path.join(dataset_dir, _id, 'images/image_{}.png'.format(_id))
            img = skimage.io.imread(self.image_path)
            height, width = img.shape
            #print(str(_id))
            self.add_image(
                "cell",
                image_id=str(_id),  # use file name as a unique image id
                path=self.image_path,
                width=width, height=height)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Get mask directory from image path
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)        

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def delete_weights(MODEL_DIR):
    weights = np.sort(glob.glob(np.sort(glob.glob(MODEL_DIR + '/*'))[-1] + '/*.h5'))
    for i in range(len(weights)-1):
        os.remove(weights[i])


if __name__ == "__main__":

    
    MODEL_DIR = "logs/"

    COCO_MODEL_PATH = "mask_rcnn_coco.h5"


    config = CellsConfig()

    """Train the model."""
    # Training dataset.
    dataset_train = CellsDataset()
    dataset_train.load_cells('train/')
    dataset_train.prepare()



    # Validation dataset
    dataset_val = CellsDataset()
    dataset_val.load_cells('validation/')
    dataset_val.prepare()

    print("Training all network")
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=30,
                augmentation = imgaug.augmenters.Sometimes(0.8, [
                imgaug.augmenters.Fliplr(0.5),
                imgaug.augmenters.Flipud(0.5),
                imgaug.augmenters.Affine(translate_px = {"x": (-2, 2), "y": (-2, 2)}),
                imgaug.augmenters.Affine(rotate = iap.Choice([90, 180, 270]))
                    ]),
                layers='all')

    delete_weights(MODEL_DIR)

