import time
import logging

import cv2
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

    def _decode_box(self, box):
        """
        Decoding boxes from [x, y, w, h] to [x1, y1, x2, y2]
        where xy1=top-left, xy2=bottom-right

        Args:
            box (np.array): Array with box coordinates

        Returns:
            (np.array): np.array with new box coordinates
        """
        y = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)

        y[:, 0] = box[:, 0] - box[:, 2] / 2  # top left box
        y[:, 1] = box[:, 1] - box[:, 3] / 2  # top left y
        y[:, 2] = box[:, 0] + box[:, 2] / 2  # bottom right box
        y[:, 3] = box[:, 1] + box[:, 3] / 2  # bottom right y

        return y

    def _letterbox(self,
                   img,
                   new_shape=(640, 640),
                   color=(114, 114, 114),
                   auto=True,
                   scaleFill=False):
        """
        Resize image in letterbox fashion
        Args:
            img (np.array): 3D numpy array of input image
            new_shape (tuple): Array with the new image height and width
            color (tuple): Color array
            auto (bool): Autoscale boolean flag
            scaleFill (bool): Scale stretch flag

        Returns:
            (np.array): np.array with the resized image
            (tuple): The height and width ratios
            (tuple): The width and height paddings
        """

        # Resize image to a 32-pixel-multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], \
                 new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], \
                    new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color)  # add border

        return img, ratio, (dw, dh)
