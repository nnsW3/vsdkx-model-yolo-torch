import time
import logging

import numpy as np
import torch
import torchvision

from vsdkx.core.interfaces import ModelDriver
from vsdkx.core.structs import Inference, FrameObject

torch.cuda.is_available = lambda: False

LOG_TAG = "Yolo Torch Driver"


class YoloTorchDriver(ModelDriver):

    def __init__(self, model_settings: dict, model_config: dict,
                 drawing_config: dict):
        super().__init__(model_settings, model_config, drawing_config)
        self._logger = logging.getLogger(LOG_TAG)

        self._input_shape = model_config['input_shape']
        self._filter_classes = model_config.get('filter_class_ids', [])
        self._classes_len = model_config['classes_len']
        self._conf_thresh = model_settings['conf_thresh']
        self._iou_thresh = model_settings['iou_thresh']
        self._device = model_settings['device']
        self._model_name = model_config.get('model_name', 'custom')
        self._model_path = {} if model_config.get('model_path') is None \
            else {'path': model_config.get('model_path')}
        self._yolo = torch.hub.load('ultralytics/yolov5',
                                    self._model_name,
                                    **self._model_path)
        self._yolo.conf = self._conf_thresh
        self._yolo.iou = self._iou_thresh

    def inference(self, frame_object: FrameObject) -> Inference:
        """
        Driver function for object detection inference

        Args:
            frame_object (FrameObject): Frame object

        Returns:
        (Inference): the result of ai
        """
        # Resize the original image for inference
        image = frame_object.frame
        self._logger.debug(
            f"frame type - {type(image)}")
        target_shape = image.shape

        predict_start = time.perf_counter()
        # Run the inference
        x = self._yolo(image, size=self._input_shape[0])
        self._logger.debug(
            f"Prediction time - {time.perf_counter() - predict_start}")

        # Run the NMS to get the boxes with the highest confidence
        y = x.pandas().xyxy[0].to_numpy()
        boxes, scores, classes = y[:, :4], y[:, 4:5], y[:, 5:6]

        if len(self._filter_classes) > 0:
            filtered_classes = list(map(lambda s: s in self._filter_classes, classes))
            boxes = list(np.array(boxes)[filtered_classes])
            scores = list(np.array(scores)[filtered_classes])
            classes = list(np.array(classes)[filtered_classes])

        return Inference(boxes, classes, scores, {})