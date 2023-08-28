#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# em: 85.5, f1: 88.4
python question-answering/run_qa_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --dataset_name squad_v2 --version_2_with_negative \
  --per_device_train_batch_size 24 --gradient_accumulation_steps 2 \
  --max_seq_length 384 --doc_stride 128 \
  --learning_rate 3e-5 --num_train_epochs 3 \
  --output_dir /home/ubuntu/ted_output/squadv2/teacher_init/ --seed 42 --mixed_precision fp16 --save_best

# em: 81.5, f1: 84.5
python question-answering/run_qa_no_trainer.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --dataset_name squad_v2 --version_2_with_negative \
  --per_device_train_batch_size 8 \
  --max_seq_length 384 --doc_stride 128 \
  --learning_rate 5e-5 --num_train_epochs 3 \
  --output_dir /home/ubuntu/ted_output/squadv2/student_init/ --seed 42 --mixed_precision fp16 --save_best