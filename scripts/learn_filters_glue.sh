#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-mnli \
    --model_type ted-deberta-v2 \
    --task_name mnli \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 2e-5 --num_train_epochs 3 \
    --num_warmup_steps 1500 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/mnli/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-mnli \
    --model_type ted-deberta-v2 \
    --task_name mnli \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 7e-5 --num_train_epochs 3 \
    --num_warmup_steps 1500 \
    --filter_interval 1  --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/mnli/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-qqp \
    --model_type ted-deberta-v2 \
    --task_name qqp \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 2e-5 --num_train_epochs 8 \
    --num_warmup_steps 5000 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/qqp/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-qqp \
    --model_type ted-deberta-v2 \
    --task_name qqp \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 8 \
    --num_warmup_steps 5000 \
    --filter_interval 1  --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/qqp/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-qnli \
    --model_type ted-deberta-v2 \
    --task_name qnli \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 1e-5 --num_train_epochs 3 \
    --num_warmup_steps 500 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/qnli/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-qnli \
    --model_type ted-deberta-v2 \
    --task_name qnli \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 5e-5 --num_train_epochs 3 \
    --num_warmup_steps 500 \
    --filter_interval 1  --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/qnli/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-sst2 \
    --model_type ted-deberta-v2 \
    --task_name sst2 \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 2e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/sst2/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-sst2 \
    --model_type ted-deberta-v2 \
    --task_name sst2 \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 4e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --filter_interval 1  --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/sst2/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-rte \
    --model_type ted-deberta-v2 \
    --task_name rte \
    --per_device_train_batch_size 16 \
    --max_length 256 \
    --learning_rate 3e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 --hidn_dropout 0.0 --att_dropout 0.0 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/rte/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-rte \
    --model_type ted-deberta-v2 \
    --task_name rte \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 --hidn_dropout 0.0 --att_dropout 0.0 \
    --filter_interval 1  --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/rte/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-cola \
    --model_type ted-deberta-v2 \
    --task_name cola \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 2e-5 --num_train_epochs 6 \
    --num_warmup_steps 100 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/cola/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-cola \
    --model_type ted-deberta-v2 \
    --task_name cola \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 \
    --filter_interval 1  --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/cola/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-mrpc \
    --model_type ted-deberta-v2 \
    --task_name mrpc \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 3e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/mrpc/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-mrpc \
    --model_type ted-deberta-v2 \
    --task_name mrpc \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 7e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --filter_interval 1 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/mrpc/student_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-stsb \
    --model_type ted-deberta-v2 \
    --task_name stsb \
    --per_device_train_batch_size 16 \
    --max_length 128 \
    --learning_rate 2e-5 --num_train_epochs 6 \
    --num_warmup_steps 100 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/stsb/teacher_stage1 --seed 42 --mixed_precision fp16

python text-classification/learn_filters_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-stsb \
    --model_type ted-deberta-v2 \
    --task_name stsb \
    --per_device_train_batch_size 16 \
    --max_length 128 \
    --learning_rate 1e-4 --num_train_epochs 6 \
    --num_warmup_steps 100 \
    --filter_interval 1 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/stsb/student_stage1 --seed 42 --mixed_precision fp16