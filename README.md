# Inference with `YoloV5`

This repository supports loading and inferencing images with differt sizes of `yolov5` both custom and pre-trained models.

### Model profile

```yaml
yolo-torch:
  classes_len: 1
  filter_class_ids:
  - 0
  input_shape:
  - 640
  - 640
  model_path: vsdkx/weights/ppl_detection_retrain_training_2.pt
 ```
 
### Model Settings

```yaml
model:
  class: vsdkx.model.yolo_torch.driver.YoloTorchDriver
  debug: false
  profile: yolo-torch
  settings:
    conf_thresh: 0.5
    device: cpu
    iou_thresh: 0.4
```

### Load different types of YoloV5 model

By default `YoloTorchDriver` is set to load `custom` trained models. When a `model_path` is specified in `profile.yaml` of the module calling the model (e.g. `PeopleDetection`), `torch.hub.load` will load the model weights from the specified location. If no `model_path` is specified, and a `model_name` is specified instead (supported options are: `yolov5s`, `yolov5m`, `yolov5l`), `torch.hub.load` will load the specified pre-trained model.

```python
self._model_name = model_config.get('model_name', 'custom') 
self._model_path = {} if model_config.get('model_path') is None \
            else {'path': model_config.get('model_path')}
self._yolo = torch.hub.load('ultralytics/yolov5',
                            self._model_name,
                            **self._model_path)
```

### Input/ Output

- Input: It receives the RGB image as a `FrameObject`:

  ```python
  def inference(self, frame_object: FrameObject) -> Inference
  ```

- Output: It returns the results of the inference (boxes, scores, classes) as an `Inference` object 
  ```python
  boxes = list(np.array(boxes)[filtered_classes])
  scores = list(np.array(scores)[filtered_classes])
  classes = list(np.array(classes)[filtered_classes])

  return Inference(boxes, classes, scores, {})
   ```
