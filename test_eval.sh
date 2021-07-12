#!/usr/bin/env bash
source activate pt_v1.1
test_epoch=-1
data_path='data'
training_set='ctw'
testing_set='ctw'
framework='PCR'
version='v0'
ct_score=0.35
type='inference_fast'  # 'inference' or 'inference_fast'. Noting that 'inference_fast' is accerlarate version for NMS

if [ "$testing_set" = "ctw" ];then
  img_dir="${data_path}/${testing_set}/test/text_image_org"
  gt_dir="${data_path}/${testing_set}/test/text_label_circum"
  mat_gt_dir="${data_path}/${testing_set}/test/text_label_curve_mat"
  eval_protocol_iou=eval_protocols/art_eval_protocol.py
  eval_protocol_deteval="eval_protocols/tot_official_eval_protocol/Python_scripts/Deteval.py"
else
  echo "Not supported testing set:${testing_set}!"
  exit
fi

yml_config_file="configs/${training_set}_snake.yaml"
det_config_file='lib/config/config.py'
snake_cfg_file='lib/utils/snake/snake_config.py'

model_dir="cvpr21_outputs/${framework}_${version}/${training_set}/model"
results="cvpr21_results/${framework}_${version}/${training_set}/${testing_set}"
vis_dir="${results}/vis_results"
det_dir="${results}/det_results"
eval_dir="${results}/eval_results"
all_eval_res_file="${results}/all_eval_res.txt"

if [ ! -d $vis_dir ]; then
  mkdir -p $vis_dir
fi
if [ ! -d $det_dir ]; then
  mkdir -p $det_dir
fi
if [ ! -d $eval_dir ]; then
  mkdir -p $eval_dir
fi

cp -r $yml_config_file $results
cp -r $det_config_file $results
cp -r $snake_cfg_file $results

test_epochs=(249)

for test_epoch in ${test_epochs[*]};do
	echo "================================testing-epoch: ${test_epoch}==================================="
	echo "================================testing-epoch: ${test_epoch}==================================">>$all_eval_res_file
if true;then
  python run.py \
          --type ${type} \
          --cfg_file ${yml_config_file} \
          --model_dir ${model_dir} \
          --test_epoch ${test_epoch} \
          --testing_set ${testing_set} \
          --results_dir ${results} \
          --det_dir ${det_dir} \
          --vis_dir ${vis_dir} \
          --gts_dir ${gt_dir} \
          demo_path ${img_dir} \
          ct_score ${ct_score}
fi
#============================================ Evaluating =================================================#
if [ "$testing_set" = "ctw" ];then
  if true;then
	rm -r ${eval_dir}
    iou_eval_res_file=${eval_dir}/iou_eval_res.txt
    eval_iou_thresh=0.5
    python ${eval_protocol_iou} \
      --imgs_dir ${img_dir} \
      --gts_dir ${gt_dir} \
      --dets_dir ${det_dir} \
      --eval_dir ${eval_dir} \
      --eval_res_file ${iou_eval_res_file} \
      --conf_thresh ${ct_score} \
      --eval_iou_thresh ${eval_iou_thresh}
    cat $iou_eval_res_file>>$all_eval_res_file
  fi
  if false;then
    dvl_eval_res_file=${eval_dir}/dvl_eval_res.txt
    rm -r $dvl_eval_res_file
    conf_thresh=${ct_score}
    python ${eval_protocol_deteval} ${det_dir} ${mat_gt_dir} ${eval_dir} ${dvl_eval_res_file} ${conf_thresh}
    cat $dvl_eval_res_file>>$all_eval_res_file
  fi
  cat $all_eval_res_file
  echo '---------------------------------------------------------------------------------------------------'
else
  echo 'Not supported testing set'
fi
done #loop for test-epochs
