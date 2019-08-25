
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
import pandas as pd
# Root directory of the project
os.environ["CUDA_VISIBLE_DEVICES"]="1"
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import gdal, ogr, os, osr
from scipy.spatial import distance
from shapely.geometry import Polygon, Point, MultiPolygon
import shapefile



class TreesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "trees"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + tree
    
    IMAGE_CHANNEL_COUNT = 3

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
    


class TreesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """


    def load_trees(self, dataset_dir, subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("tree", 1, "tree")
        
        dataset_dir = os.path.join(dataset_dir, subset+'/')
        
        # Access to dictionaty. Dictionary contain id and masks.
        self.annotations = np.load(dataset_dir + 'seg_dict.npy').item()
        
        for _id in self.annotations:
            self.image_path = os.path.join(dataset_dir, 'image_{}.jpg'.format(_id))
            img = skimage.io.imread(self.image_path)
            height, width = img.shape[:2]
            self.add_image(
                "tree",
                image_id=str(_id),  
                path=self.image_path,
                width=width, height=height)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_id = str(image_id)
        mask = []
        for element in self.annotations[image_id]:
            mask.append(self.annotations[image_id][element])
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        return self.image_path


class InferenceConfig(TreesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == "__main__":

    region_name = str(sys.argv[1])

    MODEL_DIR = "logs/{}".format(region_name)

    config = TreesConfig()

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weig

    # Either set a specific path or find last trained weights
    last_model = np.sort(glob.glob("logs/{}/*".format(region_name)))[-1]

    model_path = np.sort(glob.glob("{}/*.h5".format(last_model)))[-1]


    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    image_path = os.path.join(dataset_dir, 'image_{}.jpg'.format(image_id))
    original_image = skimage.io.imread(image_path)

    results = model.detect([original_image], verbose=1)
    r = results[0]
    
    for i, roi in enumerate(r['rois']):
        
        mask = r['masks'][:,:,i][roi[0]:roi[2], roi[1]:roi[3]].astype(int)