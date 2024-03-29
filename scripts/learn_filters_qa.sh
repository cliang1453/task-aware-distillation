#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python question-answering/learn_filters_qa_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-base-squadv2 \
    --model_type ted-deberta-v2 \
    --dataset_name squad_v2 --version_2_with_negative \
    --per_device_train_batch_size 48 \
    --max_seq_length 384 --doc_stride 128 \
    --learning_rate 3e-5 --num_train_epochs 3 \
    --filter_interval 1 \
    --output_dir /home/ubuntu/ted_output/squadv2/teacher_stage1 --seed 42 --mixed_precision fp16

python question-answering/learn_filters_qa_no_trainer.py \
    --model_name_or_path cliang1453/deberta-v3-xsmall-squadv2 \
    --model_type ted-deberta-v2 \
    --dataset_name squad_v2 --version_2_with_negative \
    --per_device_train_batch_size 8 \
    --max_seq_length 384 --doc_stride 128 \
    --learning_rate 5e-5 --num_train_epochs 3 \
    --filter_interval 1 --filter_output_dim 768 \
    --output_dir /home/ubuntu/ted_output/squadv2/student_stage1 --seed 42 --mixed_precision fp16