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
- name: output_2024-11-04 14:29:28.506521
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
      value: 0.8799019607843137
      name: Accuracy
    - type: f1
      value: 0.913884007029877
      name: F1
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_2024-11-04 14:29:28.506521

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8869
- Accuracy: 0.8799
- F1: 0.9139
- Combined Score: 0.8969

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 0
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 50.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:------:|:--------------:|
| 0.5155        | 1.0   | 459   | 0.3912          | 0.8725   | 0.9107 | 0.8916         |
| 0.517         | 2.0   | 918   | 0.3308          | 0.8676   | 0.9039 | 0.8858         |
| 0.3882        | 3.0   | 1377  | 0.3180          | 0.875    | 0.9116 | 0.8933         |
| 0.2694        | 4.0   | 1836  | 0.4245          | 0.8848   | 0.9191 | 0.9020         |
| 0.2293        | 5.0   | 2295  | 0.3783          | 0.8701   | 0.9081 | 0.8891         |
| 0.2436        | 6.0   | 2754  | 0.3495          | 0.8824   | 0.9140 | 0.8982         |
| 0.2751        | 7.0   | 3213  | 0.4805          | 0.8627   | 0.9028 | 0.8828         |
| 0.2716        | 8.0   | 3672  | 0.4212          | 0.8799   | 0.9123 | 0.8961         |
| 0.1782        | 9.0   | 4131  | 0.3841          | 0.8873   | 0.9151 | 0.9012         |
| 0.2085        | 10.0  | 4590  | 0.4476          | 0.8848   | 0.9150 | 0.8999         |
| 0.2399        | 11.0  | 5049  | 0.4791          | 0.875    | 0.9101 | 0.8925         |
| 0.1606        | 12.0  | 5508  | 0.4461          | 0.8799   | 0.9120 | 0.8960         |
| 0.1132        | 13.0  | 5967  | 0.5042          | 0.8652   | 0.9047 | 0.8849         |
| 0.1442        | 14.0  | 6426  | 0.4562          | 0.875    | 0.9128 | 0.8939         |
| 0.1699        | 15.0  | 6885  | 0.4558          | 0.8946   | 0.9231 | 0.9088         |
| 0.1629        | 16.0  | 7344  | 0.5170          | 0.8775   | 0.9120 | 0.8947         |
| 0.0684        | 17.0  | 7803  | 0.6305          | 0.8824   | 0.9152 | 0.8988         |
| 0.094         | 18.0  | 8262  | 0.6300          | 0.8775   | 0.9107 | 0.8941         |
| 0.0913        | 19.0  | 8721  | 0.5987          | 0.8701   | 0.9078 | 0.8890         |
| 0.297         | 20.0  | 9180  | 0.6047          | 0.8725   | 0.9097 | 0.8911         |
| 0.1579        | 21.0  | 9639  | 0.6155          | 0.8848   | 0.9165 | 0.9007         |
| 0.0463        | 22.0  | 10098 | 0.8098          | 0.8701   | 0.9075 | 0.8888         |
| 0.0908        | 23.0  | 10557 | 0.7219          | 0.8799   | 0.9136 | 0.8967         |
| 0.218         | 24.0  | 11016 | 0.5479          | 0.8824   | 0.9155 | 0.8989         |
| 0.0501        | 25.0  | 11475 | 0.7056          | 0.8775   | 0.9113 | 0.8944         |
| 0.283         | 26.0  | 11934 | 0.6547          | 0.8627   | 0.9034 | 0.8831         |
| 0.0182        | 27.0  | 12393 | 0.6736          | 0.8824   | 0.9143 | 0.8983         |
| 0.2112        | 28.0  | 12852 | 0.6268          | 0.8799   | 0.9130 | 0.8964         |
| 0.0668        | 29.0  | 13311 | 0.7945          | 0.8775   | 0.9113 | 0.8944         |
| 0.1135        | 30.0  | 13770 | 0.8243          | 0.8725   | 0.9097 | 0.8911         |
| 0.0831        | 31.0  | 14229 | 0.6650          | 0.8799   | 0.9133 | 0.8966         |
| 0.0091        | 32.0  | 14688 | 0.6540          | 0.8995   | 0.9264 | 0.9130         |
| 0.0443        | 33.0  | 15147 | 0.8136          | 0.8848   | 0.9171 | 0.9010         |
| 0.0989        | 34.0  | 15606 | 0.7650          | 0.8848   | 0.9162 | 0.9005         |
| 0.1002        | 35.0  | 16065 | 0.7000          | 0.8897   | 0.9183 | 0.9040         |
| 0.0389        | 36.0  | 16524 | 0.8313          | 0.8824   | 0.9149 | 0.8986         |
| 0.0266        | 37.0  | 16983 | 0.7411          | 0.8971   | 0.9242 | 0.9106         |
| 0.0332        | 38.0  | 17442 | 0.8440          | 0.8824   | 0.9149 | 0.8986         |
| 0.0668        | 39.0  | 17901 | 0.9309          | 0.8775   | 0.9123 | 0.8949         |
| 0.0901        | 40.0  | 18360 | 0.9177          | 0.8775   | 0.9120 | 0.8947         |
| 0.0316        | 41.0  | 18819 | 0.8767          | 0.8824   | 0.9152 | 0.8988         |
| 0.0188        | 42.0  | 19278 | 0.8398          | 0.8897   | 0.9209 | 0.9053         |
| 0.0004        | 43.0  | 19737 | 0.8324          | 0.8799   | 0.9142 | 0.8970         |
| 0.1429        | 44.0  | 20196 | 0.8199          | 0.8824   | 0.9152 | 0.8988         |
| 0.024         | 45.0  | 20655 | 0.9115          | 0.8799   | 0.9145 | 0.8972         |
| 0.0734        | 46.0  | 21114 | 0.9071          | 0.8799   | 0.9145 | 0.8972         |
| 0.0015        | 47.0  | 21573 | 0.8823          | 0.8799   | 0.9145 | 0.8972         |
| 0.0683        | 48.0  | 22032 | 0.8573          | 0.8824   | 0.9155 | 0.8989         |
| 0.0243        | 49.0  | 22491 | 0.8772          | 0.8848   | 0.9171 | 0.9010         |
| 0.0354        | 50.0  | 22950 | 0.8869          | 0.8799   | 0.9139 | 0.8969         |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.2.1+cu121
- Datasets 2.16.1
- Tokenizers 0.19.1