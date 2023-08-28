#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# acc: 90.6
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name mnli \
  --per_device_train_batch_size 64 \
  --max_length 256 \
  --learning_rate 2e-5 --num_train_epochs 3 \
  --num_warmup_steps 1500 \
  --output_dir /home/ubuntu/ted_output/mnli/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 88.3
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name mnli \
  --per_device_train_batch_size 64 \
  --max_length 256 \
  --learning_rate 7e-5 --num_train_epochs 3 \
  --num_warmup_steps 1500 \
  --output_dir /home/ubuntu/ted_output/mnli/student_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 92.5, f1: 89.9
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name qqp \
  --per_device_train_batch_size 64 \
  --max_length 256 \
  --learning_rate 2e-5 --num_train_epochs 8 \
  --num_warmup_steps 5000 \
  --output_dir /home/ubuntu/ted_output/qqp/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 91.7, f1: 88.9
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name qqp \
  --per_device_train_batch_size 64 \
  --max_length 256 \
  --learning_rate 6e-5 --num_train_epochs 8 \
  --num_warmup_steps 5000 \
  --output_dir /home/ubuntu/ted_output/qqp/student_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 94.1
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name qnli \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 1e-5 --num_train_epochs 3 \
  --num_warmup_steps 500 \
  --output_dir /home/ubuntu/ted_output/qnli/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 92.6
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name qnli \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 5e-5 --num_train_epochs 3 \
  --num_warmup_steps 500 \
  --output_dir /home/ubuntu/ted_output/qnli/student_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 96.4
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name sst2 \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 2e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 \
  --output_dir /home/ubuntu/ted_output/sst2/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 93.6
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name sst2 \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 4e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 \
  --output_dir /home/ubuntu/ted_output/sst2/student_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 86.3
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name rte \
  --per_device_train_batch_size 16 \
  --max_length 256 \
  --learning_rate 3e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 --cls_dropout 0.0 --hidn_dropout 0.0 --att_dropout 0.0 \
  --output_dir /home/ubuntu/ted_output/rte/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 83.4
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name rte \
  --per_device_train_batch_size 16 \
  --max_length 256 \
  --learning_rate 6e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 --cls_dropout 0.0 --hidn_dropout 0.0 --att_dropout 0.0 \
  --output_dir /home/ubuntu/ted_output/rte/student_init/ --seed 42 --mixed_precision fp16 --save_best

# mcc: 68.4
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name cola \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 2e-5 --num_train_epochs 6 \
  --num_warmup_steps 100 \
  --output_dir /home/ubuntu/ted_output/cola/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# mcc: 66.0
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name cola \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 6e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 --cls_dropout 0.0 \
  --output_dir /home/ubuntu/ted_output/cola/student_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 90.9, f1: 93.5
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name mrpc \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 3e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 \
  --output_dir /home/ubuntu/ted_output/mrpc/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# acc: 90.4, f1: 93.0
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name mrpc \
  --per_device_train_batch_size 32 \
  --max_length 256 \
  --learning_rate 7e-5 --num_train_epochs 6 \
  --num_warmup_steps 0 \
  --output_dir /home/ubuntu/ted_output/mrpc/student_init/ --seed 42 --mixed_precision fp16 --save_best

# spearmanr: 91.5
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name stsb \
  --per_device_train_batch_size 16 \
  --max_length 128 \
  --learning_rate 2e-5 --num_train_epochs 6 \
  --num_warmup_steps 100 \
  --output_dir /home/ubuntu/ted_output/stsb/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# spearmanr: 90.4
python text-classification/run_glue_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --task_name stsb \
  --per_device_train_batch_size 16 \
  --max_length 128 \
  --learning_rate 1e-4 --num_train_epochs 6 \
  --num_warmup_steps 100 \
  --output_dir /home/ubuntu/ted_output/stsb/student_init/ --seed 42 --mixed_precision fp16 --save_best