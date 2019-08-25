# Cancer-Cell-Segmentation
Cancer cell segmentation based on Image Processing and Deep learning.
The preprocess.py implements Otsu's method to separate nuclei from cell bodies. After a some edge filters, Watershed techniques segmentates each cellular body in the laboratory sample. Each one of them is saved in a train, validation and test folder that will feed a Mask-RCNN model for cellular segmentation.
Before training the Mask-RCNN, you need to download the COCO weights and store it in the same folder.
Run preprocess.py, then training.py and you can test the model with predict.py

