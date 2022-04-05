import unittest
import numpy as np
from numpy.core.fromnumeric import shape

from vsdkx.core.structs import FrameObject, Inference
from vsdkx.model.yolo_torch.driver import YoloTorchDriver


class TestDriver(unittest.TestCase):
    MODEL_DRIVER = None
    
    def test_driver_construction(self):
        
        model_config = {
            "input_shape": (640, 640),
            "filter_class_ids": [16],
            "classes_len": 1,
            "model_name": "yolov5m",
        }
        model_settings = {
            "conf_thresh": 0.5,
            "iou_thresh": 0.7,
            "device": "cpu"
        }
        
        TestDriver.MODEL_DRIVER = YoloTorchDriver(model_settings, model_config, drawing_config={})
        
        self.assertIsInstance(TestDriver.MODEL_DRIVER, YoloTorchDriver)

    def test_driver_inference(self):
        shape =  TestDriver.MODEL_DRIVER._input_shape       
        shape = shape + (3,)

        frame = np.zeros(shape)
        frame_object = FrameObject(frame, {})

        inference = TestDriver.MODEL_DRIVER.inference(frame_object)
        
        self.assertIsInstance(inference, Inference)

        self.assertEqual(len(inference.boxes), 0)
        self.assertEqual(len(inference.classes), 0)
        self.assertEqual(len(inference.scores), 0)

        self.assertEqual(inference.extra, {})

