#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# acc: 88.8
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-mnli-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-mnli-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name mnli \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 7e-5 --num_train_epochs 3 \
    --num_warmup_steps 1500 \
    --kl_alpha 5 --mse_alpha 7500 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/mnli/ted --seed 42 --mixed_precision fp16 --save_best

# acc: 92.2, f1: 89.5
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-qqp-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-qqp-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name qqp \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 8 \
    --num_warmup_steps 5000 \
    --kl_alpha 20 --mse_alpha 1000 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/qqp/ted --seed 42 --mixed_precision fp16 --save_best

# acc: 93.1
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-qnli-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-qnli-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name qnli \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 5e-5 --num_train_epochs 3 \
    --num_warmup_steps 500 \
    --kl_alpha 20 --mse_alpha 100 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/qnli/ted --seed 42 --mixed_precision fp16 --save_best

# acc: 94.4
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-sst2-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name sst2 \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 4e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --kl_alpha 20 --mse_alpha 20 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/sst2/ted --seed 42 --mixed_precision fp16 --save_best

# acc: 83.0
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-rte-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-rte-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name rte \
    --per_device_train_batch_size 16 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 --hidn_dropout 0.0 --att_dropout 0.0 \
    --kl_alpha 20 --mse_alpha 50 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/rte/ted --seed 42 --mixed_precision fp16 --save_best

mcc: 68.3
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-cola-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-cola-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name cola \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 \
    --kl_alpha 0 --mse_alpha 10 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/cola/ted --seed 42 --mixed_precision fp16 --save_best

# acc: 91.7, f1: 93.9
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-mrpc-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-mrpc-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name mrpc \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 7e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --kl_alpha 20 --mse_alpha 100 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/mrpc/ted --seed 42 --mixed_precision fp16 --save_best

# spearmanr: 91.1
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-stsb-student-stage1 --teacher_model_name_or_path cliang1453/deberta-v3-base-stsb-teacher-stage1 \
    --model_type ted-deberta-v2 \
    --task_name stsb \
    --per_device_train_batch_size 16 \
    --max_length 128 \
    --learning_rate 1e-4 --num_train_epochs 6 \
    --num_warmup_steps 100 \
    --kl_alpha 20 --mse_alpha 50 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/stsb/ted --seed 42 --mixed_precision fp16 --save_best