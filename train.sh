#!/bin/bash
model="efficientnet" #"efficientnet" - "efficientnet_pretrained_b{}"4,5
depth_coefficient=1.0
width_coefficient=1.0
resolution_coefficient=1.0
compound_coefficient=1.0
# Size efficientnet https://github.com/lukemelas/EfficientNet-PyTorch/issues/42
img_size=256 # Initial 256
crop_size=224 # Initial 224
epochs=300
freezed_epochs=0
batch_size=32 # efficientnet -> 32 / efficientnet_pretrained_b4 -> 16
optimizer="sgd"
snapshot=1
lr=0.01 # learning_rate
path_extension="steplr_phase3_CustomDA_CoefficientSearch"
model_path="results/"$model"_"$optimizer"_lr"$lr"_d"$depth_coefficient"_w"$width_coefficient"_r"$resolution_coefficient"_c"$compound_coefficient

# --pretrained_imagenet
python3 -u train.py --optimizer $optimizer --model_name $model --learning_rate $lr --epochs $epochs \
                    --depth_coefficient $depth_coefficient --width_coefficient $width_coefficient \
                    --resolution_coefficient $resolution_coefficient --compound_coefficient $compound_coefficient \
                    --output_dir $model_path --batch_size $batch_size --path_extension $path_extension --data_augmentation \
                    --depth_coefficient $depth_coefficient --width_coefficient $width_coefficient \
                    --resolution_coefficient $resolution_coefficient --freezed_epochs $freezed_epochs \
                    --img_size $img_size --crop_size $crop_size --snapshot $snapshot
