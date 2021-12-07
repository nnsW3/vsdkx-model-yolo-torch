import time

import cv2
import numpy as np
import torch
import torchvision
from vsdkx.core.interfaces import ModelDriver
from vsdkx.core.structs import Inference, FrameObject

torch.cuda.is_available = lambda: False


class YoloTorchDriver(ModelDriver):

    def __init__(self, model_settings: dict, model_config: dict,
                 drawing_config: dict):
        super().__init__(model_settings, model_config, drawing_config)
        self._input_shape = model_config['input_shape']
        self._filter_classes = model_config.get('filter_class_ids', [])
        self._classes_len = model_config['classes_len']
        self._conf_thresh = model_settings['conf_thresh']
        self._iou_thresh = model_settings['iou_thresh']
        self._device = model_settings['device']
        self._yolo = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path=model_config['model_path'])

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
        target_shape = image.shape
        resized_image = self._resize_img(image, self._input_shape)

        inf_start = time.perf_counter()
        # Run the inference
        x = self._yolo(resized_image)

        # Run the NMS to get the boxes with the highest confidence
        y = self._process_pred(x)
        boxes, scores, classes = [], [], []

        for pred in y:
            boxes, scores, classes = pred[:, :4], pred[:, 4:5], pred[:, 5:6]
            boxes = self._scale_boxes(boxes, self._input_shape,
                                      target_shape)

        result_boxes = []
        result_scores = []
        result_classes = []
        if len(self._filter_classes) > 0:
            # Go through the prediction results
            for box, score, c_id in zip(boxes, scores, classes):
                # Iterate over the predicted bounding boxes and filter
                #   the boxes with class "person"
                if c_id in self._filter_classes:
                    result_boxes.append(box)
                    result_scores.append(score)
                    result_classes.append(c_id)
        else:
            result_boxes = boxes
            result_scores = scores
            result_classes = classes

        return Inference(result_boxes, result_classes, result_scores, {})

    def _scale_boxes(self, boxes, input_shape, target_shape):
        """
        Scales the boxes to the size of the target image

        Args:
            boxes (np.array): Array containing the bounding boxes
            input_shape (tuple): The shape of the resized image
            target_shape (tuple): The shape of the target image

        Returns:
            (np.array): np.array with the scaled bounding boxes
        """
        gain = min(input_shape[0] / target_shape[0],
                   input_shape[1] / target_shape[1])
        pad = (input_shape[1] - target_shape[1] * gain) / 2, \
              (input_shape[0] - target_shape[0] * gain) / 2
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :] /= gain

        return boxes

    def _resize_img(self, image, input_shape):
        """
        Resize input image to the expected input shape

        Args:
            image (np.array): 3D numpy array of input image
            input_shape (tuple): The shape of the input image

        Returns:
            (array): Resized image
        """

        image_resized = self._letterbox(image,
                                        new_shape=input_shape[0], auto=False)[
            0]
        image_np = image_resized[:, :, ::-1].transpose(2, 0, 1)
        image_np = np.ascontiguousarray(image_np)
        image_np = torch.from_numpy(image_np).to(self._device)
        image_np = image_np.float()  # uint8 to fp16/32
        image_np = image_np / 255.0
        image_np = image_np.unsqueeze(0)

        return image_np

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

    def _process_pred(self, prediction):
        """
        Processes the prediction results and passes them through NMS

        Args:
            prediction (np.array): Array with the post-processed
            inference predictions

        Returns:
             (np.array): np.array detections with shape
              nx6 (x1, y1, x2, y2, conf, cls)
        """
        # Get candidates with confidence higher than the threshold
        xc = prediction[..., 4] > self._conf_thresh  # candidates

        # Maximum width and height
        max_wh = 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        multi_label = (prediction.shape[2] - 5) > 1
        output = [torch.zeros((0, 6), device=prediction.device)] * \
                 prediction.shape[0]

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center box, center y, width, height) to (x1, y1, x2, y2)
            box = self._decode_box(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > self._conf_thresh).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()),
                              1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > self._conf_thresh]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[
                      :max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * max_wh  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, self._iou_thresh)  # NMS

            output[xi] = x[i].detach().cpu().numpy()

        return output

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
