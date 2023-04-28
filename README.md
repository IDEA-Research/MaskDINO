Mask DINO <img src="figures/dinosaur.png" width="30">
========
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/panoptic-segmentation-on-coco-test-dev)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-test-dev?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=mask-dino-towards-a-unified-transformer-based-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-dino-towards-a-unified-transformer-based-1/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=mask-dino-towards-a-unified-transformer-based-1)


[Feng Li*](https://fengli-ust.github.io/), [Hao Zhang*](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=zh-CN), [Huaizhe Xu](https://scholar.google.com/citations?user=zgaTShsAAAAJ&hl=en&scioq=Huaizhe+Xu), [Shilong Liu](https://www.lsl.zone/), [Lei Zhang](https://scholar.google.com/citations?hl=zh-CN&user=fIlGZToAAAAJ), [Lionel M. Ni](https://scholar.google.com/citations?hl=zh-CN&user=OzMYwDIAAAAJ), and [Heung-Yeung Shum](https://scholar.google.com.hk/citations?user=9akH-n8AAAAJ&hl=en).

This repository is the official implementation of the [Mask DINO: Towards A Unified Transformer-based
Framework for Object Detection and Segmentation](https://arxiv.org/abs/2206.02777) (DINO pronounced `daɪnoʊ' as in dinosaur). Our code is based on [detectron2](https://github.com/facebookresearch/detectron2). [detrex](https://github.com/IDEA-Research/detrex) version is opensource simultaneously.

:fire: We release a strong open-set object detection and segmentation model [OpenSeeD](https://arxiv.org/pdf/2303.08131.pdf) based on MaskDINO that achieves the best results on open-set object segmentation tasks. Code and checkpoints are available [here](https://github.com/IDEA-Research/OpenSeeD).

<details close>
<summary> <font size=8><strong>News</strong></font> </summary>


[2023/2] Mask DINO has been accepted to CVPR 2023!

[2022/9] We release a toolbox [**detrex**](https://github.com/IDEA-Research/detrex) that provides state-of-the-art Transformer-based detection algorithms. It includes DINO **with better performance** and Mask DINO will also be released with detrex implementation. Welcome to use it! </br>
  - Supports Now: [DETR](https://arxiv.org/abs/2005.12872), [Deformble DETR](https://arxiv.org/abs/2010.04159), [Conditional DETR](https://arxiv.org/abs/2108.06152), [Group-DETR](https://arxiv.org/abs/2207.13085), [DAB-DETR](https://arxiv.org/abs/2201.12329), [DN-DETR](https://arxiv.org/abs/2203.01305), [DINO](https://arxiv.org/abs/2203.03605).

[2022/7] Code for [DINO](https://arxiv.org/pdf/2203.03605.pdf) is available [here](https://github.com/IDEACVR/DINO)!

[2022/5][DN-DETR](https://arxiv.org/abs/2203.01305) is accepted to CVPR 2022 as an **Oral** presentation. Code is now avaliable [here](https://github.com/IDEA-opensource/DN-DETR).

[2022/4][DAB-DETR](https://arxiv.org/abs/2201.12329) is accepted to ICLR 2022. Code is now avaliable [here](https://github.com/IDEA-opensource/DAB-DETR).

[2022/3]We release a SOTA detection model [DINO](https://arxiv.org/abs/2203.03605) that for the first time establishes a DETR-like model as a SOTA model on the [leaderboard](https://paperswithcode.com/sota/object-detection-on-coco). Code will be avaliable [here](https://github.com/IDEACVR/DINO).

[2022/3]We build a repo [Awesome Detection Transformer](https://github.com/IDEACVR/awesome-detection-transformer) to present papers about transformer for detection and segmentation. Welcome to your attention!
</details>


### Features 

* A **unified** architecture for object detection, panoptic, instance and semantic segmentation.
* Achieve **task and data cooperation** between detection and segmentation.
* **State-of-the-art** performance under the same setting.
* Support major detection and segmentation datasets: COCO, ADE20K, Cityscapes.


### Code Updates

* [2022/12/02] Our code and checkpoints are available! Mask DINO further Achieves <strong>51.7</strong> and <strong>59.0</strong> box AP on COCO with a ResNet-50 and SwinL without extra detection data, **outperforming DINO** under the same setting!

* [2022/6] We propose a unified detection and segmentation model [Mask DINO](https://arxiv.org/pdf/2206.02777.pdf) that achieves the best results on all the three segmentation tasks (**54.7** AP on [COCO instance leaderboard](https://paperswithcode.com/sota/instance-segmentation-on-coco), **59.5** PQ on [COCO panoptic leaderboard](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-test-dev), and **60.8** mIoU on [ADE20K semantic leaderboard](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k))!

<details open>
<summary> <font size=8><strong>Todo list</strong></font> </summary>

- [x] Release code and checkpoints
  
- [ ] Release model conversion checkpointer from DINO to MaskDINO
 
- [ ] Release GPU cluster submit scripts based on submitit for multi-node training
 
- [ ] Release EMA training for large models
 
- [ ] Release more large models
</details>



***

## Installation

See [installation instructions](INSTALL.md).



## Getting Started
See [Inference Demo with Pre-trained Model](demo/README.md)

See [Results](#results).

See [Preparing Datasets for MaskDINO](datasets/README.md).

See [Getting Started](#getting-started-1).

See [More Usage](#more-usage).

![MaskDINO](figures/framework.jpg)

***

# Results
In this part, we present the clean models that do not use extra detection data or tricks.
### COCO Instance Segmentation and Object Detection
 we follow DINO to use hidden dimension `2048` in the encoder of feedforward by default. We also use the mask-enhanced
box initialization proposed in our paper in instance segmentation and detection. To better present our model, we also list the models trained with 
hidden dimension `1024` (`hid 1024`) and not using mask-enhance initialization (`no mask enhance`) in this table.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Mask AP</th>
<th valign="bottom">Box AP</th>
<th valign="bottom">Params</th>
<th valign="bottom">GFlops</th>
<th valign="bottom">download</th>

 <tr><td align="left">MaskDINO (hid 1024) | <a href="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml">config</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">46.1</td>
<td align="center">51.5</td>
<td align="center">47M</td>
<td align="center">226</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth">model</a></td>
</tr>

 <tr><td align="left">MaskDINO | <a href="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml">config</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">46.3</td>
<td align="center">51.7</td>
<td align="center">52M</td>
<td align="center">286</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth">model</a></td>
</tr>

 <tr><td align="left">MaskDINO (no mask enhance) | <a href="configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml">config</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">50</td>
<td align="center">52.1</td>
<td align="center">58.3</td>
<td align="center">223</td>
<td align="center">1326</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth">model</a></td>
</tr>

 <tr><td align="left">MaskDINO | <a href="configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml">config</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">50</td>
<td align="center">52.3</td>
<td align="center">59.0</td>
<td align="center">223</td>
<td align="center">1326</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth">model</a></td>
</tr>
 <tr><td align="left">MaskDINO+O365 data+1.2 x larger image</td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">20</td>
<td align="center">54.5</td>
<td align="center">---</td>
<td align="center">223</td>
<td align="center">1326</td>
<td align="center">To Release</td>
</tr>
</tbody></table>

### COCO Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">epochs</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Mask AP</th>
<th valign="bottom">Box AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">download</th>

 <tr><td align="left">MaskDINO | <a href="configs/coco/panoptic-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml">config</a></td>
<td align="center">R50</td>
<td align="center">50</td>
<td align="center">53.0</td>
<td align="center">48.8</td>
<td align="center">44.3</td>
<td align="center">60.6</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth">model</a></td>

 <tr><td align="left">MaskDINO | <a href="configs/coco/panoptic-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml">config</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">50</td>
<td align="center">58.3</td>
<td align="center">50.6</td>
<td align="center">56.2</td>
<td align="center">67.5</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth">model</a></td>
</tr>

 <tr><td align="left">MaskDINO+O365 data+1.2 x larger image</td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">20</td>
<td align="center">59.4</td>
<td align="center">53.0</td>
<td align="center">57.7</td>
<td align="center">67.3</td>
<td align="center">To Release</td>
</tr>

</tbody></table>

### Semantic Segmentation
We use hidden dimension `1024` and 100 queries for semantic segmentation.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">download</th>

 <tr><td align="left">MaskDINO | <a href="configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml">config</a></td>
<td align="center">ADE20K</td>
<td align="center">R50</td>
<td align="center">160k</td>
<td align="center">48.7</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth">model</a></td>

 <tr><td align="left">MaskDINO | <a href="configs/cityscapes/semantic-segmentation/maskdino_R50_bs16_90k_steplr.yaml">config</a></td>
<td align="center">Cityscapes</td>
<td align="center">R50</td>
<td align="center">90k</td>
<td align="center">79.8</td>
<td align="center"><a href="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth">model</a></td>

</tbody></table>

You can also find all these models [here](https://github.com/IDEA-Research/detrex-storage/releases/tag/maskdino-v0.1.0).

All models were trained with  **4** NVIDIA A100 GPUs (ResNet-50 based models) or **8** NVIDIA A100 GPUs (Swin-L based models).

We will release more pretrained models in the future.
# Getting Started

In the above tables, the "Name" column contains a link `config_path` to the config file, and the corresponding model checkpoints
can be downloaded from the link in `model`.

If your dataset files are not under this repo, you need to add `export DETECTRON2_DATASETS=/path/to/your/data` or use Symbolic Link `ln -s`
to link the dataset into this repo before the
following command first.
#### Evalaluate our pretrained models
* You can download our pretrained models and evaluate them with the following commands.
  ```sh
  python train_net.py --eval-only --num-gpus 8 --config-file config_path MODEL.WEIGHTS /path/to/checkpoint_file
  ```
  for example, to reproduce our instance segmentation result, you can copy the config path from the table, download the pretrained checkpoint into `/path/to/checkpoint_file`, and run 
  ```sh
  python train_net.py --eval-only --num-gpus 8 --config-file configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml MODEL.WEIGHTS /path/to/checkpoint_file
  ```
  which can reproduce the model. 
#### Train MaskDINO to reproduce results
* Use the above command without `eval-only` will train the model. For Swin backbones, you need to specify the path of the pretrained backbones with `MODEL.WEIGHTS /path/to/pretrained_checkpoint`
    ```sh
  python train_net.py --num-gpus 8 --config-file config_path MODEL.WEIGHTS /path/to/checkpoint_file
  ```
* For ResNet-50 models, training on 8 GPU requires around `15G` memory on each GPU and `3` days training for 50 epochs. 
* For Swin-L models, training on 8 gpu required memory `60G` on each GPU. If your gpu do not have enough 
  memory, you may also train with 16 GPUs with distributed training on two nodes.
* We use total batch size 16 for all our models. If train on 1 GPU, you need to figure out learning rate and batch size by yourself
    ```sh
  python train_net.py --num-gpus 1 --config-file config_path SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE
  ```

You can also refer to [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


# More Usage

### Mask-enhanced box initialization

We provide 2 ways to convert predicted masks to boxes to initialize decoder boxes. You can set as follows
* `MODEL.MaskDINO.INITIALIZE_BOX_TYPE: no` not using mask enhanced box initialization
* `MODEL.MaskDINO.INITIALIZE_BOX_TYPE: mask2box`  a fast conversion way
* `MODEL.MaskDINO.INITIALIZE_BOX_TYPE: bitmask`  provided conversion from detectron2, slower but more accurate conversion. 

These two conversion ways do not affect the final performance much, you can choose either way. 

In addition, if you already
train a model for 50 epochs without mask-enhance box initialization, you can plug in this method and simply 
finetune the model in the last few epochs (i.e., load from 32K iteration trained model and finetune it). This way can
also achieve similar performance compared with training from scratch, but more flexible.

### Model components
MaskDINO  consists of three components: a backbone, a pixel decoder and a Transformer decoder.
You can easily replace each of these three components with your own implementation.

* **backbone**: Define and register your backbone under `maskdino/modeling/backbone`. You can follow the Swin Transformer as an example.
  
* **pixel decoder**: pixel decoder is actually the multi-scale encoder in DINO and Deformable DETR, we follow mask2former to call
  it pixel decoder. It is in `maskdino/modeling/pixel_decoder`, you can change your multi-scale encoder. The returned values 
  include 
  1. `mask_features` is the per-pixel embeddings with resolution 1/4 of the original image, obtained by fusing backbone 1/4 features and multi-scale encoder encoded 1/8 features. This is used to produce binary masks.
  2. `multi_scale_features`, which is the multi-scale inputs to the Transformer decoder.
  For ResNet-50 models with 4 scales, we use resolution 1/32, 1/16, and 1/8 but you can use arbitrary resolutions here, and follow DINO to additionally downsample
     1/32 to get a 4th scale with 1/64 resolution. For 5-scale models with SwinL, we additional use 1/4 resolution features as in DINO.

* **transformer decoder**: it mainly follows DINO decoder to do detection and segmentation tasks. It is defined in `maskdino/modeling/transformer_decoder`.


## LICNESE
Mask DINO is released under the Apache 2.0 license. Please see the [LICENSE](LICNESE) file for more information.

Copyright (c) IDEA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## <a name="CitingMaskDINO"></a>Citing Mask DINO

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```BibTeX
@misc{li2022mask,
      title={Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation}, 
      author={Feng Li and Hao Zhang and Huaizhe xu and Shilong Liu and Lei Zhang and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2206.02777},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{li2022dn,
      title={Dn-detr: Accelerate detr training by introducing query denoising},
      author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13619--13627},
      year={2022}
}

@inproceedings{
      liu2022dabdetr,
      title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
      author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}
```

## Acknowledgement

Many thanks to these excellent opensource projects 
* [Mask2Former](https://github.com/facebookresearch/Mask2Former) 
* [DINO](https://github.com/IDEA-Research/DINO)

