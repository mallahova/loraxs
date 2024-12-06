---
language:
- en
license: mit
library_name: peft
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
- f1
base_model: roberta-large
model-index:
- name: output_2024-11-05 18:48:26.328306
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: GLUE MRPC
      type: glue
      args: mrpc
    metrics:
    - type: accuracy
      value: 0.8946078431372549
      name: Accuracy
    - type: f1
      value: 0.9238938053097344
      name: F1
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_2024-11-05 18:48:26.328306

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8781
- Accuracy: 0.8946
- F1: 0.9239
- Combined Score: 0.9093

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 12
- eval_batch_size: 8
- seed: 0
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:--------------:|
| 0.1651        | 1.0   | 306  | 1.0764          | 0.8676   | 0.9078 | 0.8877         |
| 0.1216        | 2.0   | 612  | 0.6604          | 0.9069   | 0.9319 | 0.9194         |
| 0.0644        | 3.0   | 918  | 0.9141          | 0.8897   | 0.9215 | 0.9056         |
| 0.0917        | 4.0   | 1224 | 0.6773          | 0.8873   | 0.9190 | 0.9031         |
| 0.0336        | 5.0   | 1530 | 0.9259          | 0.875    | 0.9116 | 0.8933         |
| 0.0437        | 6.0   | 1836 | 0.8066          | 0.8725   | 0.9094 | 0.8910         |
| 0.005         | 7.0   | 2142 | 0.9084          | 0.8922   | 0.9225 | 0.9073         |
| 0.0073        | 8.0   | 2448 | 0.8324          | 0.8775   | 0.9129 | 0.8952         |
| 0.0058        | 9.0   | 2754 | 0.7981          | 0.8922   | 0.9217 | 0.9069         |
| 0.0024        | 10.0  | 3060 | 0.8781          | 0.8946   | 0.9239 | 0.9093         |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.2.1+cu121
- Datasets 2.16.1
- Tokenizers 0.19.1