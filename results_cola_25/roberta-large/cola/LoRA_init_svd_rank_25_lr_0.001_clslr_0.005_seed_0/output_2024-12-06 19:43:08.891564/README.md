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
- matthews_correlation
base_model: roberta-large
model-index:
- name: output_2024-12-06 19:43:08.891564
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: GLUE COLA
      type: glue
      args: cola
    metrics:
    - type: matthews_correlation
      value: 0.6138859958577028
      name: Matthews Correlation
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_2024-12-06 19:43:08.891564

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the GLUE COLA dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6061
- Matthews Correlation: 0.6139

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
- num_epochs: 50.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Matthews Correlation |
|:-------------:|:-----:|:-----:|:---------------:|:--------------------:|
| 0.3501        | 1.0   | 713   | 0.5347          | 0.3710               |
| 0.7656        | 2.0   | 1426  | 0.4743          | 0.5757               |
| 1.2079        | 3.0   | 2139  | 1.0137          | 0.0                  |
| 1.0497        | 4.0   | 2852  | 0.4873          | 0.5368               |
| 1.3593        | 5.0   | 3565  | 1.0409          | 0.4655               |
| 1.1657        | 6.0   | 4278  | 0.9633          | 0.5333               |
| 0.1017        | 7.0   | 4991  | 1.4661          | 0.2645               |
| 0.2349        | 8.0   | 5704  | 0.7538          | 0.5516               |
| 1.0473        | 9.0   | 6417  | 0.8280          | 0.5417               |
| 0.3432        | 10.0  | 7130  | 0.7302          | 0.5937               |
| 0.0182        | 11.0  | 7843  | 0.5199          | 0.5935               |
| 0.0001        | 12.0  | 8556  | 1.4449          | 0.5193               |
| 0.7625        | 13.0  | 9269  | 0.6280          | 0.5334               |
| 1.1517        | 14.0  | 9982  | 1.1371          | 0.5209               |
| 0.4285        | 15.0  | 10695 | 0.5836          | 0.6057               |
| 0.7953        | 16.0  | 11408 | 0.7744          | 0.5946               |
| 0.2802        | 17.0  | 12121 | 0.6915          | 0.5941               |
| 0.2201        | 18.0  | 12834 | 0.8690          | 0.4066               |
| 1.9445        | 19.0  | 13547 | 0.6238          | 0.6489               |
| 0.1167        | 20.0  | 14260 | 0.4377          | 0.6381               |
| 0.8086        | 21.0  | 14973 | 1.4222          | 0.5345               |
| 0.0146        | 22.0  | 15686 | 0.9237          | 0.5883               |
| 0.062         | 23.0  | 16399 | 1.3447          | 0.3907               |
| 0.0551        | 24.0  | 17112 | 0.8871          | 0.5822               |
| 0.1061        | 25.0  | 17825 | 1.0291          | 0.5570               |
| 0.4303        | 26.0  | 18538 | 0.6799          | 0.5996               |
| 0.002         | 27.0  | 19251 | 0.6725          | 0.6282               |
| 0.6872        | 28.0  | 19964 | 0.5619          | 0.6745               |
| 0.3286        | 29.0  | 20677 | 0.4038          | 0.6506               |
| 1.1017        | 30.0  | 21390 | 0.7502          | 0.5866               |
| 0.6849        | 31.0  | 22103 | 1.1866          | 0.5755               |
| 0.1615        | 32.0  | 22816 | 0.4679          | 0.6649               |
| 0.0854        | 33.0  | 23529 | 0.7976          | 0.5765               |
| 0.1389        | 34.0  | 24242 | 0.8682          | 0.5920               |
| 0.1005        | 35.0  | 24955 | 0.8486          | 0.6238               |
| 0.0085        | 36.0  | 25668 | 0.6227          | 0.6484               |
| 0.0576        | 37.0  | 26381 | 0.6398          | 0.6207               |
| 0.6536        | 38.0  | 27094 | 0.9326          | 0.5826               |
| 0.0023        | 39.0  | 27807 | 0.6559          | 0.6579               |
| 0.5649        | 40.0  | 28520 | 0.7589          | 0.6029               |
| 0.0146        | 41.0  | 29233 | 0.6183          | 0.5943               |
| 0.5056        | 42.0  | 29946 | 0.5893          | 0.6408               |
| 1.0894        | 43.0  | 30659 | 0.5642          | 0.6293               |
| 0.031         | 44.0  | 31372 | 0.6876          | 0.6227               |
| 0.0076        | 45.0  | 32085 | 0.6641          | 0.6054               |
| 0.0013        | 46.0  | 32798 | 0.8570          | 0.6098               |
| 0.8038        | 47.0  | 33511 | 0.6374          | 0.6238               |
| 0.011         | 48.0  | 34224 | 0.7494          | 0.6139               |
| 0.2636        | 49.0  | 34937 | 0.5299          | 0.6356               |
| 0.1957        | 50.0  | 35650 | 0.6061          | 0.6139               |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.2.1+cu121
- Datasets 2.16.1
- Tokenizers 0.19.1