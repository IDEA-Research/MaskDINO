We use ResNet and Swin as the backbone in our model.

* `convert-torchvision-to-d2.py`

Tool to convert torchvision pre-trained weights for D2.

```
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
python tools/convert-torchvision-to-d2.py resnet101-63fe2227.pth R-101.pkl
```

* `convert-pretrained-swin-model-to-d2.py`

Tool to convert Swin Transformer pre-trained weights for D2.

```
pip install timm

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_small_patch4_window7_224.pth swin_small_patch4_window7_224.pkl

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pkl

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_large_patch4_window12_384_22k.pth swin_large_patch4_window12_384_22k.pkl
```


* `analyze_model.py`

Tool to analyze model parameters and flops.

Usage for semantic segmentation (ADE20K only, use with caution!):

```
python tools/analyze_model.py --num-inputs 1 --tasks flop --use-fixed-input-size --config-file CONFIG_FILE
```

Note that, for semantic segmentation (ADE20K only), we use a dummy image with fixed size that equals to `cfg.INPUT.CROP.SIZE[0] x cfg.INPUT.CROP.SIZE[0]`.
Please do not use `--use-fixed-input-size` for calculating FLOPs on other datasets like Cityscapes!

Usage for panoptic and instance segmentation:

```
python tools/analyze_model.py --num-inputs 100 --tasks flop --config-file CONFIG_FILE
```

Note that, for panoptic and instance segmentation, we compute the average flops over 100 real validation images.
