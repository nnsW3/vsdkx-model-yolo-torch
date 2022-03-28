import unittest

from vsdkx.model.yolo_torch.driver import YoloTorchDriver


class TestDriver(unittest.TestCase):
    
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
        
        model_driver = YoloTorchDriver(model_settings, model_config, drawing_config={})
        
        self.assertIsInstance(model_driver, YoloTorchDriver)
