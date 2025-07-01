#!/usr/bin/env bash

# Make it executable: chmod +x run_all.sh
# Then run: ./run_all.sh

python main.py -net samed2 -exp_name DATASET_NAME \
-sam_ckpt ./checkpoints/latest_epoch_0217.pth \
-sam_config sam2_hiera_s \
-image_size 1024 -out_size 1024 -b 4 \
-data_path DATA_ROOT_PATH \
-val_file_dir TEST_FILE_PATH \
