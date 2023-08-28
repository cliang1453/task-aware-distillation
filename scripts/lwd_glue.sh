#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# acc: 88.4
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-mnli \
    --model_type ted-deberta-v2 \
    --task_name mnli \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 7e-5 --num_train_epochs 3 \
    --num_warmup_steps 1500 \
    --kl_alpha 5 --mse_alpha 100 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/mnli/lwd --seed 42 --mixed_precision fp16 --save_best

# acc: 91.8, f1: 89.0
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-qqp \
    --model_type ted-deberta-v2 \
    --task_name qqp \
    --per_device_train_batch_size 64 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 8 \
    --num_warmup_steps 5000 \
    --kl_alpha 20 --mse_alpha 20 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/qqp/lwd --seed 42 --mixed_precision fp16 --save_best
    
# acc: 92.9
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-qnli \
    --model_type ted-deberta-v2 \
    --task_name qnli \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 5e-5 --num_train_epochs 3 \
    --num_warmup_steps 500 \
    --kl_alpha 20 --mse_alpha 10 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/qnli/lwd --seed 42 --mixed_precision fp16 --save_best

# acc: 94.3
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-sst2 \
    --model_type ted-deberta-v2 \
    --task_name sst2 \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 4e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --kl_alpha 20 --mse_alpha 20 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/sst2/lwd --seed 42 --mixed_precision fp16 --save_best

# acc: 80.8
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-rte \
    --model_type ted-deberta-v2 \
    --task_name rte \
    --per_device_train_batch_size 16 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 --hidn_dropout 0.0 --att_dropout 0.0 \
    --kl_alpha 20 --mse_alpha 20 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/rte/lwd --seed 42 --mixed_precision fp16 --save_best

# mcc: 65.8
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-cola \
    --model_type ted-deberta-v2 \
    --task_name cola \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 6e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 --cls_dropout 0.0 \
    --kl_alpha 0 --mse_alpha 5 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/cola/lwd --seed 42 --mixed_precision fp16 --save_best

# acc: 90.9, f1: 93.4
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-mrpc \
    --model_type ted-deberta-v2 \
    --task_name mrpc \
    --per_device_train_batch_size 32 \
    --max_length 256 \
    --learning_rate 7e-5 --num_train_epochs 6 \
    --num_warmup_steps 0 \
    --kl_alpha 20 --mse_alpha 5 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/mrpc/lwd --seed 42 --mixed_precision fp16 --save_best

# spearmanr: 90.6
python text-classification/ted_glue_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-stsb \
    --model_type ted-deberta-v2 \
    --task_name stsb \
    --per_device_train_batch_size 16 \
    --max_length 128 \
    --learning_rate 1e-4 --num_train_epochs 6 \
    --num_warmup_steps 100 \
    --kl_alpha 20 --mse_alpha 5 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/stsb/lwd --seed 42 --mixed_precision fp16 --save_best