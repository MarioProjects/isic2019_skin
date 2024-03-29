#!/bin/bash
# color-densenet-40-k -> 40: net depth / k: growth rate -> Tipycal 40-12 or 40-48 
model="resnetd101b" #"efficientnet" - "efficientnet_pretrained_b{}"4,5 - "seresnext50_32x4d" - "resnetd101b"
depth_coefficient=1.0
width_coefficient=1.0
resolution_coefficient=1.0
compound_coefficient=1.0
# Size efficientnet https://github.com/lukemelas/EfficientNet-PyTorch/issues/42
img_size=256 # Initial 256
crop_size=224 # Initial 224
epochs=300
freezed_epochs=0
batch_size=32 # efficientnet -> 32 / efficientnet_pretrained_b4 -> 16 / resnetd101b 32
optimizer="sgd"
snapshot=1
lr=0.01 # learning_rate
path_extension="steplr_phase4_weightedLossDivided_NormalSteps"

if test "model" = "efficientnet"
then
    model_path="results/"$model"_"$optimizer"_lr"$lr"_d"$depth_coefficient"_w"$width_coefficient"_r"$resolution_coefficient"_c"$compound_coefficient
else
    model_path="results/"$model"_"$optimizer"_lr"$lr
fi

# efficientenet --> depth_coefficient $depth_coefficient --width_coefficient $width_coefficient \
#                    --resolution_coefficient $resolution_coefficient --compound_coefficient $compound_coefficient \

# --pretrained_imagenet / --retinex / --shade_of_gray / --colornet / --cutmix / --freezed_epochs $freezed_epochs

python3 -u train.py --optimizer $optimizer --model_name $model --learning_rate $lr --epochs $epochs \
                    --output_dir $model_path --batch_size $batch_size --path_extension $path_extension --data_augmentation \
                    --img_size $img_size --crop_size $crop_size --snapshot $snapshot --weighted_loss
