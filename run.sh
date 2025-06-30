#!/usr/bin/env bash

# Make it executable: chmod +x run_all.sh
# Then run: ./run_all.sh

# python main.py -net sam -exp_name DATASET_NAME \
# -sam_ckpt ./checkpoints/sam_vit_b_01ec64.pth \
# -image_size 1024 -out_size 1024 -b 4 \
# -data_path DATA_ROOT_PATH \
# -val_file_dir TEST_FILE_PATH \

# python main.py -net medsam -exp_name DATASET_NAME \
# -sam_ckpt ./checkpoints/medsam_vit_b.pth \
# -image_size 1024 -out_size 1024 -b 4 \
# -data_path DATA_ROOT_PATH \
# -val_file_dir TEST_FILE_PATH \
  
# python main.py -net medsam2 -exp_name DATASET_NAME \
# -sam_ckpt ./checkpoints/MedSAM2_pretrain.pth \
# -sam_config sam2_hiera_t_original \
# -image_size 1024 -out_size 1024 -b 4 \
# -data_path DATA_ROOT_PATH \
# -val_file_dir TEST_FILE_PATH \
  
# python main.py -net sam2 -exp_name DATASET_NAME \
# -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
# -sam_config sam2_hiera_s_original \
# -image_size 1024 -out_size 1024 -b 4 \
# -data_path DATA_ROOT_PATH \
# -val_file_dir TEST_FILE_PATH \

python main.py -net samed2 -exp_name DATASET_NAME \
-sam_ckpt ./checkpoints/latest_epoch_0217.pth \
-sam_config sam2_hiera_s \
-image_size 1024 -out_size 1024 -b 4 \
-data_path DATA_ROOT_PATH \
-val_file_dir TEST_FILE_PATH \
  
