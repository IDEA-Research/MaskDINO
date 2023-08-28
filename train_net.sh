CUDA_VISIBLE_DEVICES=0,1,2 python train_net.py \
--num-gpus 3 --config-file /MaskDINO/configs/coco/instance-segmentation/luggage_parts.yaml \
MODEL.WEIGHTS /MaskDINO/runs/luggage_parts_1508_2/model_final.pth \
OUTPUT_DIR runs/luggage_parts_1508_extendepoch \
INPUT.IMAGE_SIZE 800 SOLVER.IMS_PER_BATCH 9 \
SOLVER.MAX_ITER 1500 SOLVER.STEPS [9999,12999]
# SOLVER.MAX_ITER 1000 SOLVER.STEPS [8888,9629]
  