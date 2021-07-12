#!/usr/bin/env bash
source activate pt_v1.1
training_set='ctw'
framework='PCR'
version='v0'
yml_config_file="configs/${training_set}_snake.yaml"
det_config_file='lib/config/config.py'
snake_cfg_file='lib/utils/snake/snake_config.py'

outputs="cvpr21_outputs/${framework}_${version}/${training_set}"
model_dir="${outputs}/model"
record_dir="${outputs}/record"
model_name="${training_set}_snake"

if [ ! -d $outputs ]; then
	mkdir -p $outputs
fi

cp -r $yml_config_file $outputs
cp -r $det_config_file $outputs
cp -r $snake_cfg_file $outputs

start_time=`date "+%Y-%m-%d %H:%M:%S"`
python train_net.py \
    --model_dir ${model_dir} \
    --record_dir ${record_dir} \
    --model ${model_name} \
    --cfg_file ${yml_config_file}
end_time=`date "+%Y-%m-%d %H:%M:%S"`
echo 'start_time:'$start_time',end_time:'$end_time
