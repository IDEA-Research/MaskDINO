## Getting Started with MaskDINO

This document provides a brief intro of the usage of **MaskDINO**.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Inference Demo with Pre-trained Models

1. Pick a model and its config file
- for example
   - config file at `/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml`.
   - Model file [MaskDINO (hid 1024) ](https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth)
2. We provide `demo.py` that is able to demo builtin configs. 
3. Run it with:
```
cd demo/
python demo.py --config-file /configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS /path/to/model_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


