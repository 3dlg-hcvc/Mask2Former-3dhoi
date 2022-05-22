import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import Instances

import numpy as np
import cv2
import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config

input_image = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_6.11/valid/7292-0-1-0+bg2.png"
SCORE_THRESHOLD = 0.5

if __name__ == "__main__":

    setup_logger()
    setup_logger(name="mask2former")

    coco_metadata = MetadataCatalog.get("coco_2017_val")

    im = cv2.imread(input_image)

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # Use the instance segmentation config file
    cfg.merge_from_file("configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    # Use the fully trained model for instance segmentation on COCO
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl'
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # Set a score threshold
    predictions = outputs["instances"]
    filter_predictions = Instances(predictions.image_size)
    flag = False
    for index in range(len(predictions)):
        score = predictions[index].scores[0]
        if score > SCORE_THRESHOLD:
            if flag == False:
                filter_predictions = predictions[index]
                flag = True
            else:
                filter_predictions = Instances.cat([filter_predictions, predictions[index]])

    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(filter_predictions.to("cpu")).get_image()

    cv2.imwrite("output.png", instance_result)