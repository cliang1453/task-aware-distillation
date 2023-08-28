# Less is More: Task-aware Layer-wise Distillation for Language Model Compression

This repo contains the code for our paper ["Less is More: Task-aware Layer-wise Distillation for Language Model Compression"](https://arxiv.org/abs/2210.01351) (ICML2023). We propose TED, a task-aware layerwise distillation approach for language model compression.

## News

**[Aug 27, 2023]** We have released the code for GLUE experiments (including KD and LWD baselines) upon requests. All teacher and student checkpoints for GLUE and SQuAD 2.0 have been released. Please submit an issue if you need examples for other tasks and models.

## Getting Started

1. Pull and run docker </br>
   ```pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel```
2. Install requirements </br>
   ```
   cd task-aware-distillation
   pip install -r requirements.txt
   ```
## GLUE

### Preparation (Optional)
Given a teacher model and a student model, we first fine-tune them on the target task if they have not yet been fine-tuned. You can run
```
./scripts/finetune_glue.sh
```
We will then use these fine-tuned models to initialize the teacher and student in Stage I. Alternatively, you may directly use the fine-tuned models we released on Huggingface model hub: 
| Task  | DeBERTaV3-base | DeBERTaV3-xs |
|-------|-----------------|--------------|
| MNLI  | 90.6 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-mnli) | 88.3 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-mnli)|
| QQP   | 89.9 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-qqp)  | 88.9 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-qqp) |
| QNLI  | 94.1 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-qnli) | 92.6 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-qnli)|
| SST-2 | 96.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-sst2) | 93.6 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-sst2)|
| RTE   | 86.3 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-rte)  | 83.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-rte) |
| CoLA  | 68.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-cola) | 66.0 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-cola)|
| MRPC  | 93.5 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-mrpc) | 93.0 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-mrpc)|
| STS-B | 91.5 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-stsb) | 90.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-stsb)|

### Stage I: Learning Task-aware Filters (Optional)
Given properly initialized teacher and student models, we add a set of layerwise task-aware filters to each of them. We train the filters on the target task to extract the task-relevant knowledge, while keeping the model frozen. You can run
```
./scripts/learn_filters_glue.sh
```
Stage I has several important hyperparameters:
- ```--filter_interval```: The layer interval between consecutive filters. Default to be one. If it is not necessary to learn a filter for every layer (e.g., a 12-layer teacher only needs 6 filters to match with a 6-layer student), you may specify a larger ```filter_interval``` to save the training cost (e.g., 12/6 = 2).
- ```--filter_output_dim```: The output dimension of each filter. Default to be the hidden dimension of the model. If the teacher and the student models have different hidden dimensions, you may specify a single ```filter_output_dim``` for both of them for dimension matching.
- ```--filter_nonlinear```: A boolean value of whether to add nonlinearities in filters. Default to be false.

We will then use the models equipped with learned filters to initialize the teacher and student in Stage II. Alternatively, you may directly use the models with learned filters we released:
| Task  | DeBERTaV3-base (Stage I) | DeBERTaV3-xs (Stage I) |
|-------|-----------------|--------------|
| MNLI  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-mnli-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-mnli-student-stage1)|
| QQP   | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-qqp-teacher-stage1)  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-qqp-student-stage1) |
| QNLI  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-qnli-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-qnli-student-stage1)|
| SST-2 | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-sst2-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-sst2-student-stage1)|
| RTE   | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-rte-teacher-stage1)  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-rte-student-stage1) |
| CoLA  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-cola-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-cola-student-stage1)|
| MRPC  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-mrpc-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-mrpc-student-stage1)|
| STS-B | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-stsb-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-stsb-student-stage1)|

### Stage II: Task-aware Distillation
Finally, we conduct task-aware distillation (TED). We distill the student model and its filters to match its filtered output representations to those of the teacher model, while keeping the teacher model and its filters frozen. You can run
```
./scripts/ted_glue.sh
```
In Stage II, there are two new hyperparameters:
- ```--kl_alpha```: The weighting of the KL-divergence loss between the teacher's and the student's final-layer output predictions. 
- ```--mse_alpha```: The weighting of the MSE loss between the output representations of the teacher's and the student's filters. The student is optimized based on ```pred_loss + kl_alpha * kl_loss + mse_alpha * layer_averaged_mse_loss```.

The final task-aware distilled student checkpoints have been released at: 
| Task  | DeBERTaV3-xs (Stage II) |
|-------|-----------------|
| MNLI  | 88.8 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-mnli-student-stage2)|
| QQP   | 89.5 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-qqp-student-stage2) |
| QNLI  | 93.1 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-qnli-student-stage2)|
| SST-2 | 94.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-sst2-student-stage2)|
| RTE   | 83.0 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-rte-student-stage2) |
| CoLA  | 68.3 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-cola-student-stage2)|
| MRPC  | 93.9 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-mrpc-student-stage2)|
| STS-B | 91.1 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-stsb-student-stage2)|

### Other Baselines
Our codebase also supports vanilla knowledge distillation (KD) and layerwise distillation (LWD) methods. You can run KD with
```
./scripts/kd_glue.sh
```
by adding ```--filter_disabled``` in the argument list and setting ```--mse_alpha = 0```. You can run LWD with
```
./scripts/lwd_glue.sh
```
by adding ```--filter_disabled``` in the argument list. 

</br>

## SQuAD 2.0

### TED
We first properly intialize the teacher and the student models on the target task by running:
```
./scripts/finetune_qa.sh
```
Alternatively, you may directly use the fine-tuned models we released:

| Task  | DeBERTaV3-base | DeBERTaV3-xs |
|-------|-----------------|--------------|
| SQuAD 2.0  | 88.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-squadv2) | 84.5 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-squadv2)|

For Stage I, we learn filters by running:
```
./scripts/learn_filters_qa.sh
```
Alternatively, you can directly use the models with learned filters we released:  

| Task  | DeBERTaV3-base (Stage I) | DeBERTaV3-xs (Stage I) |
|-------|-----------------|--------------|
| SQuAD 2.0  | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-base-squadv2-teacher-stage1) | [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-squadv2-student-stage1)|

For Stage II, we conduct task-aware distillation by running:
```
./scripts/ted_qa.sh
```
The final task-aware distilled student checkpoints have been released at:

| Task  | DeBERTaV3-xs (Stage II) |
|-------|--------------|
| SQuAD 2.0  | 86.4 [[ckpt]](https://huggingface.co/cliang1453/deberta-v3-xsmall-squadv2-student-stage2)|

### Other Baselines
Our codebase also supports vanilla knowledge distillation (KD) and layerwise distillation (LWD) methods. You can run:
```
./scripts/kd_qa.sh
./scripts/lwd_qa.sh
```

## Citation
```
@article{liang2022less,
  title={Less is More: Task-aware Layer-wise Distillation for Language Model Compression},
  author={Liang, Chen and Zuo, Simiao and Zhang, Qingru and He, Pengcheng and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2210.01351},
  year={2022}
}
```

## Contact Information
Please submit a GitHub issue **if you need examples for other tasks and models**, or other issues related to this package. For questions related to this paper, please contact Chen Liang (cliang73@gatech.edu).