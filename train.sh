#!/bin/bash
model="efficientnet_pretrained_b4" #"efficientnet"
depth_coefficient=1.0
width_coefficient=1.0
resolution_coefficient=1.0
compound_coefficient=1.0
# Size efficientnet https://github.com/lukemelas/EfficientNet-PyTorch/issues/42
img_size=415
crop_size=380
epochs=300
freezed_epochs=135
batch_size=16
optimizer="sgd"
lr=0.01 # learning_rate
path_extension="steplr_phase3_CustomDA_CoefficientSearch"
model_path="results/"$model"_"$optimizer"_lr"$lr"_d"$depth_coefficient"_w"$width_coefficient"_r"$resolution_coefficient"_c"$compound_coefficient

python3 -u train.py --optimizer $optimizer --model_name $model --learning_rate $lr --epochs $epochs \
                    --depth_coefficient $depth_coefficient --width_coefficient $width_coefficient \
                    --resolution_coefficient $resolution_coefficient --compound_coefficient $compound_coefficient \
                    --output_dir $model_path --batch_size $batch_size --path_extension $path_extension --data_augmentation \
                    --depth_coefficient $depth_coefficient --width_coefficient $width_coefficient \
                    --resolution_coefficient $resolution_coefficient --freezed_epochs $freezed_epochs \
                    --img_size $img_size --crop_size $crop_size --pretrained_imagenet
