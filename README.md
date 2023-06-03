# Less is More: Task-aware Layer-wise Distillation for Language Model Compression

This repo contains the code for our paper ["Less is More: Task-aware Layer-wise Distillation for Language Model Compression"](https://arxiv.org/abs/2210.01351) (ICML2023). We propose TED, a task-aware layerwise distillation approach for language model compression.

</br>

## Getting Started

1. Pull and run docker </br>
   ```pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel```
2. Install requirements </br>
   ```
   cd task-aware-distillation
   pip install -r requirements.txt
   ```
   
## Preparation

Given a teacher model and a student model, we first train them on the target task if they have not been trained yet. Here we provide an example of fine-tuning a DeBERTaV3-base model and a DeBERTaV3-xsmall model on the SQuAD 2.0 task:
```
./scripts/finetune.sh
```
We then use the fine-tuned DeBERTaV3-base model and the fine-tuned DeBERTaV3-xsmall model to initialized the teacher model and the student model in Stage I, respectively.

</br>

## Stage I: Learning Task-aware Filters

Given a properly initialized teacher/student model, we introduce a set of layerwise task-aware filters. We train the filters on the target task to extract the task-relevant knowledge, while keeping the model frozen. Here we provide an example of training the filters for the DeBERTaV3-base teacher and the DeBERTaV3-xsmall student on the SQuAD 2.0 task:
```
./scripts/learn_filters.sh
```
Stage I has several important hyperparameters:
- ```--filter_interval```: The layer interval between consecutive filters. Default to be one. If it is not necessary to learn a filter for every layer (e.g., a 12-layer teacher only needs 6 filters to match with a 6-layer student), you may specify a larger ```filter_interval``` to save the training cost (e.g., 12/6 = 2).
- ```--filter_output_dim```: The output dimension of each filter. Default to be the hidden dimension of the model. If the teacher and the student models have different hidden dimensions, you may specify a single ```filter_output_dim``` for both of them for dimension matching.
- ```--filter_nonlinear```: A boolean value of whether to add nonlinearities in filters. Default to be false.

Stage I training takes around 40 minutes for both the DeBERTaV3-base teacher and the DeBERTaV3-xsmall student under fp16 using a single Nvidia V100 GPU.

</br>

## Stage II: Task-aware Distillation

Finally, we conduct task-aware distillation (TED). We distill the student model and its filters to match its filtered output representations to those of the teacher model, while keeping the teacher model and its filters frozen. Here is an example of conducting TED between the DeBERTaV3-base teacher and the DeBERTaV3-xsmall student on the SQuAD 2.0 task:
```
./scripts/ted.sh
```
In Stage II, there are two new hyperparameters:
- ```--kl_alpha```: The weighting of the KL-divergence loss between the teacher's and the student's final-layer output predictions. 
- ```--mse_alpha```: The weighting of the MSE loss between the output representations of the teacher's and the student's filters. The student is optimized based on ```pred_loss + kl_alpha * kl_loss + mse_alpha * layer_averaged_mse_loss```.

After Stage II, the student should achieve a validation performance around:
```
    "exact": 83.00345321317275,
    "f1": 85.87542003616846,
```
Stage II takes around 14 hours under fp16 using a single Nvidia V100 GPU.

</br>

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