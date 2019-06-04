#!/bin/bash
model="efficientnet"
depth_coefficient=1.0
width_coefficient=1.0
resolution_coefficient=1.0
compound_coefficient=1.0
epochs=200
batch_size=32
optimizer="adam"
lr=0.001 # learning_rate
model_path="results/"$model"_"$optimizer"_"$lr"_d"$depth_coefficient"_w"$width_coefficient"_r"$resolution_coefficient"_c"$compound_coefficient"/"

python3 -u train.py --optimizer $optimizer --model_name $model --learning_rate $lr --epochs $epochs \
                    --depth_coefficient $depth_coefficient --width_coefficient $width_coefficient \
                    --resolution_coefficient $resolution_coefficient --compound_coefficient $compound_coefficient \
                    --output_dir $model_path --batch_size $batch_size