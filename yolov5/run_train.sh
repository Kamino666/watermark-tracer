CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
--img 640 --batch 16 --epochs 3 \
--data ../data/WatermarkDataset/watermark.yaml \
--weights yolov5l.pt