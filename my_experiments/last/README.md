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
- name: output_2024-12-03 21:39:44.432967
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
      value: 0.8553921568627451
      name: Accuracy
    - type: f1
      value: 0.8984509466437176
      name: F1
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_2024-12-03 21:39:44.432967

This model is a fine-tuned version of [roberta-large](https://huggingface.co/roberta-large) on the GLUE MRPC dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9116
- Accuracy: 0.8554
- F1: 0.8985
- Combined Score: 0.8769
- Accuracy 10: 0.8529
- Accuracy 11: 0.8505
- Accuracy 12: 0.8529
- Accuracy 13: 0.8505
- Accuracy 14: 0.8505
- Accuracy 15: 0.8554
- Accuracy 16: 0.8554
- Accuracy 17: 0.8578
- Accuracy 18: 0.8554
- Accuracy 19: 0.8529
- Accuracy 20: 0.8578
- Accuracy 21: 0.8529
- Accuracy 22: 0.8554
- Accuracy 23: 0.8554
- Accuracy 24: 0.8554
- Accuracy 25: 0.8554

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
- train_batch_size: 11
- eval_batch_size: 8
- seed: 0
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 70.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1     | Combined Score | Accuracy 10 | Accuracy 11 | Accuracy 12 | Accuracy 13 | Accuracy 14 | Accuracy 15 | Accuracy 16 | Accuracy 17 | Accuracy 18 | Accuracy 19 | Accuracy 20 | Accuracy 21 | Accuracy 22 | Accuracy 23 | Accuracy 24 | Accuracy 25 |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:------:|:--------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 0.6142        | 1.0   | 334   | 0.5017          | 0.7426   | 0.8346 | 0.7886         | 0.7451      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      | 0.7426      |
| 0.4385        | 2.0   | 668   | 0.4102          | 0.8456   | 0.8919 | 0.8688         | 0.8456      | 0.8480      | 0.8505      | 0.8529      | 0.8505      | 0.8480      | 0.8529      | 0.8480      | 0.8505      | 0.8505      | 0.8529      | 0.8529      | 0.8554      | 0.8529      | 0.8456      | 0.8456      |
| 0.4452        | 3.0   | 1002  | 0.3186          | 0.8456   | 0.8865 | 0.8660         | 0.8554      | 0.8529      | 0.8578      | 0.8554      | 0.8578      | 0.8480      | 0.8554      | 0.8529      | 0.8529      | 0.8505      | 0.8505      | 0.8554      | 0.8554      | 0.8505      | 0.8529      | 0.8505      |
| 0.3582        | 4.0   | 1336  | 0.3522          | 0.8701   | 0.9091 | 0.8896         | 0.875       | 0.875       | 0.8775      | 0.8824      | 0.8701      | 0.8725      | 0.8701      | 0.8701      | 0.8701      | 0.8701      | 0.8775      | 0.8775      | 0.8799      | 0.8799      | 0.875       | 0.8725      |
| 0.3972        | 5.0   | 1670  | 0.3267          | 0.8652   | 0.9053 | 0.8853         | 0.8578      | 0.8578      | 0.8554      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8603      | 0.8627      | 0.8603      | 0.8578      | 0.8603      | 0.8603      | 0.8603      | 0.8627      |
| 0.3681        | 6.0   | 2004  | 0.3148          | 0.8578   | 0.8957 | 0.8768         | 0.8652      | 0.8652      | 0.8676      | 0.8603      | 0.8578      | 0.8603      | 0.8603      | 0.8603      | 0.8603      | 0.8627      | 0.8627      | 0.8652      | 0.8603      | 0.8578      | 0.8578      | 0.8554      |
| 0.2455        | 7.0   | 2338  | 0.4084          | 0.8652   | 0.9063 | 0.8857         | 0.8554      | 0.8578      | 0.8578      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8627      | 0.8627      | 0.8627      | 0.8627      | 0.8627      | 0.8578      | 0.8578      | 0.8578      | 0.8627      |
| 0.3553        | 8.0   | 2672  | 0.4535          | 0.8529   | 0.8997 | 0.8763         | 0.8505      | 0.8505      | 0.8554      | 0.8578      | 0.8578      | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8554      | 0.8554      | 0.8529      | 0.8529      | 0.8505      | 0.8529      | 0.8529      |
| 0.2922        | 9.0   | 3006  | 0.3254          | 0.8456   | 0.8869 | 0.8662         | 0.8431      | 0.8431      | 0.8505      | 0.8456      | 0.8407      | 0.8456      | 0.8480      | 0.8480      | 0.8456      | 0.8480      | 0.8456      | 0.8431      | 0.8480      | 0.8456      | 0.8480      | 0.8431      |
| 0.2909        | 10.0  | 3340  | 0.3566          | 0.8529   | 0.8940 | 0.8735         | 0.8578      | 0.8603      | 0.8603      | 0.8627      | 0.8627      | 0.8480      | 0.8505      | 0.8480      | 0.8480      | 0.8505      | 0.8529      | 0.8529      | 0.8505      | 0.8529      | 0.8529      | 0.8529      |
| 0.3042        | 11.0  | 3674  | 0.4374          | 0.8456   | 0.8934 | 0.8695         | 0.8431      | 0.8431      | 0.8431      | 0.8382      | 0.8407      | 0.8407      | 0.8407      | 0.8456      | 0.8431      | 0.8431      | 0.8456      | 0.8407      | 0.8456      | 0.8456      | 0.8407      | 0.8431      |
| 0.2753        | 12.0  | 4008  | 0.4434          | 0.8456   | 0.8927 | 0.8691         | 0.8603      | 0.8603      | 0.8603      | 0.8529      | 0.8529      | 0.8505      | 0.8505      | 0.8505      | 0.8529      | 0.8529      | 0.8529      | 0.8480      | 0.8431      | 0.8505      | 0.8529      | 0.8456      |
| 0.2404        | 13.0  | 4342  | 0.4534          | 0.8652   | 0.9079 | 0.8865         | 0.8603      | 0.8603      | 0.8627      | 0.8652      | 0.8627      | 0.8627      | 0.8652      | 0.8676      | 0.8676      | 0.8701      | 0.8701      | 0.8652      | 0.8652      | 0.8652      | 0.8652      | 0.8652      |
| 0.3065        | 14.0  | 4676  | 0.3976          | 0.8652   | 0.9043 | 0.8848         | 0.8505      | 0.8554      | 0.8603      | 0.8554      | 0.8578      | 0.8603      | 0.8578      | 0.8627      | 0.8578      | 0.8652      | 0.8627      | 0.8554      | 0.8529      | 0.8554      | 0.8603      | 0.8554      |
| 0.3092        | 15.0  | 5010  | 0.4313          | 0.8603   | 0.9032 | 0.8818         | 0.8627      | 0.8627      | 0.8652      | 0.8603      | 0.8603      | 0.8652      | 0.8652      | 0.8603      | 0.8578      | 0.8603      | 0.8627      | 0.8554      | 0.8603      | 0.8603      | 0.8578      | 0.8578      |
| 0.2683        | 16.0  | 5344  | 0.4008          | 0.8578   | 0.9003 | 0.8791         | 0.8505      | 0.8456      | 0.8578      | 0.8554      | 0.8554      | 0.8554      | 0.8603      | 0.8627      | 0.8627      | 0.8627      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8554      |
| 0.1511        | 17.0  | 5678  | 0.5149          | 0.8603   | 0.9022 | 0.8813         | 0.8554      | 0.8578      | 0.8603      | 0.8578      | 0.8627      | 0.8652      | 0.8676      | 0.8578      | 0.8578      | 0.8578      | 0.8652      | 0.8578      | 0.8603      | 0.8554      | 0.8578      | 0.8603      |
| 0.2883        | 18.0  | 6012  | 0.4066          | 0.8554   | 0.8974 | 0.8764         | 0.8554      | 0.8578      | 0.8578      | 0.8627      | 0.8652      | 0.8627      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8603      | 0.8603      | 0.8554      | 0.8578      | 0.8578      | 0.8578      |
| 0.2665        | 19.0  | 6346  | 0.4552          | 0.8676   | 0.9069 | 0.8873         | 0.8603      | 0.8603      | 0.8554      | 0.8578      | 0.8603      | 0.8578      | 0.8603      | 0.8627      | 0.8652      | 0.8652      | 0.8627      | 0.8652      | 0.8652      | 0.8627      | 0.8627      | 0.8652      |
| 0.2353        | 20.0  | 6680  | 0.4332          | 0.8578   | 0.9014 | 0.8796         | 0.8627      | 0.8578      | 0.8627      | 0.8627      | 0.8578      | 0.8578      | 0.8603      | 0.8554      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8529      | 0.8578      | 0.8529      | 0.8505      |
| 0.2157        | 21.0  | 7014  | 0.4090          | 0.8578   | 0.9014 | 0.8796         | 0.8603      | 0.8554      | 0.8627      | 0.8627      | 0.8603      | 0.8676      | 0.8652      | 0.8627      | 0.8578      | 0.8578      | 0.8554      | 0.8578      | 0.8578      | 0.8578      | 0.8603      | 0.8603      |
| 0.2533        | 22.0  | 7348  | 0.4530          | 0.8529   | 0.8969 | 0.8749         | 0.8578      | 0.8603      | 0.8603      | 0.8554      | 0.8554      | 0.8554      | 0.8578      | 0.8554      | 0.8529      | 0.8529      | 0.8505      | 0.8480      | 0.8480      | 0.8480      | 0.8505      | 0.8529      |
| 0.245         | 23.0  | 7682  | 0.4723          | 0.8578   | 0.8986 | 0.8782         | 0.8554      | 0.8529      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8603      | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8554      | 0.8578      | 0.8578      | 0.8578      |
| 0.2505        | 24.0  | 8016  | 0.5171          | 0.8529   | 0.8966 | 0.8747         | 0.8456      | 0.8431      | 0.8480      | 0.8456      | 0.8480      | 0.8480      | 0.8578      | 0.8505      | 0.8505      | 0.8554      | 0.8529      | 0.8505      | 0.8529      | 0.8529      | 0.8529      | 0.8578      |
| 0.2344        | 25.0  | 8350  | 0.5207          | 0.8554   | 0.8988 | 0.8771         | 0.8480      | 0.8505      | 0.8529      | 0.8505      | 0.8505      | 0.8578      | 0.8529      | 0.8554      | 0.8578      | 0.8554      | 0.8554      | 0.8529      | 0.8529      | 0.8529      | 0.8554      | 0.8554      |
| 0.2894        | 26.0  | 8684  | 0.5185          | 0.8480   | 0.8970 | 0.8725         | 0.8554      | 0.8554      | 0.8603      | 0.8603      | 0.8603      | 0.8627      | 0.8554      | 0.8603      | 0.8603      | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8554      | 0.8554      |
| 0.1651        | 27.0  | 9018  | 0.5428          | 0.8603   | 0.9002 | 0.8802         | 0.8529      | 0.8529      | 0.8529      | 0.8554      | 0.8554      | 0.8578      | 0.8627      | 0.8603      | 0.8554      | 0.8578      | 0.8603      | 0.8554      | 0.8578      | 0.8554      | 0.8554      | 0.8578      |
| 0.2582        | 28.0  | 9352  | 0.5320          | 0.8603   | 0.9022 | 0.8813         | 0.8529      | 0.8529      | 0.8529      | 0.8529      | 0.8505      | 0.8578      | 0.8627      | 0.8652      | 0.8627      | 0.8627      | 0.8603      | 0.8627      | 0.8578      | 0.8554      | 0.8529      | 0.8554      |
| 0.2156        | 29.0  | 9686  | 0.5538          | 0.8505   | 0.8954 | 0.8729         | 0.8529      | 0.8529      | 0.8578      | 0.8554      | 0.8529      | 0.8529      | 0.8529      | 0.8505      | 0.8505      | 0.8480      | 0.8480      | 0.8505      | 0.8529      | 0.8554      | 0.8505      | 0.8505      |
| 0.2144        | 30.0  | 10020 | 0.5207          | 0.8603   | 0.9022 | 0.8813         | 0.8578      | 0.8578      | 0.8652      | 0.8578      | 0.8578      | 0.8652      | 0.8603      | 0.8603      | 0.8529      | 0.8603      | 0.8578      | 0.8603      | 0.8554      | 0.8578      | 0.8603      | 0.8627      |
| 0.2477        | 31.0  | 10354 | 0.5220          | 0.8554   | 0.8970 | 0.8762         | 0.8529      | 0.8554      | 0.8554      | 0.8505      | 0.8505      | 0.8480      | 0.8505      | 0.8529      | 0.8480      | 0.8529      | 0.8505      | 0.8505      | 0.8480      | 0.8505      | 0.8480      | 0.8505      |
| 0.2713        | 32.0  | 10688 | 0.5145          | 0.8480   | 0.8924 | 0.8702         | 0.8603      | 0.8554      | 0.8603      | 0.8603      | 0.8554      | 0.8603      | 0.8578      | 0.8578      | 0.8554      | 0.8480      | 0.8529      | 0.8456      | 0.8456      | 0.8456      | 0.8480      | 0.8480      |
| 0.2088        | 33.0  | 11022 | 0.6093          | 0.8480   | 0.8927 | 0.8704         | 0.8505      | 0.8505      | 0.8529      | 0.8505      | 0.8456      | 0.8456      | 0.8480      | 0.8480      | 0.8431      | 0.8431      | 0.8456      | 0.8456      | 0.8480      | 0.8480      | 0.8480      | 0.8480      |
| 0.132         | 34.0  | 11356 | 0.7828          | 0.8554   | 0.8995 | 0.8774         | 0.8529      | 0.8505      | 0.8529      | 0.8529      | 0.8529      | 0.8554      | 0.8554      | 0.8554      | 0.8603      | 0.8554      | 0.8554      | 0.8554      | 0.8529      | 0.8554      | 0.8505      | 0.8554      |
| 0.1397        | 35.0  | 11690 | 0.6359          | 0.8505   | 0.8950 | 0.8727         | 0.8554      | 0.8529      | 0.8529      | 0.8529      | 0.8505      | 0.8554      | 0.8554      | 0.8529      | 0.8554      | 0.8529      | 0.8505      | 0.8456      | 0.8480      | 0.8505      | 0.8505      | 0.8505      |
| 0.1749        | 36.0  | 12024 | 0.6368          | 0.8554   | 0.8991 | 0.8773         | 0.8554      | 0.8554      | 0.8603      | 0.8529      | 0.8554      | 0.8529      | 0.8480      | 0.8480      | 0.8480      | 0.8529      | 0.8529      | 0.8554      | 0.8554      | 0.8529      | 0.8505      | 0.8529      |
| 0.1524        | 37.0  | 12358 | 0.6803          | 0.8480   | 0.8946 | 0.8713         | 0.8529      | 0.8554      | 0.8529      | 0.8529      | 0.8480      | 0.8480      | 0.8505      | 0.8456      | 0.8480      | 0.8505      | 0.8505      | 0.8505      | 0.8480      | 0.8480      | 0.8480      | 0.8480      |
| 0.1338        | 38.0  | 12692 | 0.6494          | 0.8480   | 0.8931 | 0.8706         | 0.8505      | 0.8505      | 0.8505      | 0.8505      | 0.8554      | 0.8554      | 0.8554      | 0.8578      | 0.8529      | 0.8505      | 0.8529      | 0.8505      | 0.8480      | 0.8505      | 0.8480      | 0.8480      |
| 0.2049        | 39.0  | 13026 | 0.5891          | 0.8627   | 0.9021 | 0.8824         | 0.8627      | 0.8578      | 0.8603      | 0.8554      | 0.8578      | 0.8603      | 0.8554      | 0.8554      | 0.8529      | 0.8554      | 0.8554      | 0.8505      | 0.8554      | 0.8529      | 0.8529      | 0.8505      |
| 0.1508        | 40.0  | 13360 | 0.7282          | 0.8505   | 0.8954 | 0.8729         | 0.8578      | 0.8529      | 0.8578      | 0.8529      | 0.8480      | 0.8456      | 0.8529      | 0.8505      | 0.8554      | 0.8505      | 0.8505      | 0.8480      | 0.8480      | 0.8480      | 0.8480      | 0.8505      |
| 0.1548        | 41.0  | 13694 | 0.7769          | 0.8554   | 0.8991 | 0.8773         | 0.8603      | 0.8627      | 0.8627      | 0.8603      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8554      | 0.8578      | 0.8578      | 0.8554      | 0.8554      | 0.8554      | 0.8554      |
| 0.179         | 42.0  | 14028 | 0.6771          | 0.8554   | 0.8974 | 0.8764         | 0.8627      | 0.8603      | 0.8578      | 0.8627      | 0.8603      | 0.8554      | 0.8603      | 0.8529      | 0.8578      | 0.8554      | 0.8578      | 0.8554      | 0.8554      | 0.8554      | 0.8578      | 0.8554      |
| 0.1456        | 43.0  | 14362 | 0.7998          | 0.8505   | 0.8950 | 0.8727         | 0.8529      | 0.8603      | 0.8578      | 0.8554      | 0.8529      | 0.8554      | 0.8603      | 0.8529      | 0.8603      | 0.8554      | 0.8603      | 0.8529      | 0.8505      | 0.8529      | 0.8505      | 0.8505      |
| 0.1269        | 44.0  | 14696 | 0.7370          | 0.8603   | 0.9009 | 0.8806         | 0.8603      | 0.8652      | 0.8652      | 0.8627      | 0.8652      | 0.8652      | 0.8627      | 0.8627      | 0.8627      | 0.8676      | 0.8603      | 0.8578      | 0.8652      | 0.8578      | 0.8578      | 0.8603      |
| 0.1938        | 45.0  | 15030 | 0.7383          | 0.8603   | 0.9009 | 0.8806         | 0.8554      | 0.8554      | 0.8578      | 0.8529      | 0.8529      | 0.8603      | 0.8652      | 0.8603      | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8554      | 0.8554      | 0.8554      |
| 0.1917        | 46.0  | 15364 | 0.8454          | 0.8578   | 0.9017 | 0.8798         | 0.8529      | 0.8652      | 0.8529      | 0.8529      | 0.8480      | 0.8529      | 0.8578      | 0.8529      | 0.8578      | 0.8627      | 0.8652      | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8554      |
| 0.082         | 47.0  | 15698 | 0.8431          | 0.8578   | 0.9014 | 0.8796         | 0.8603      | 0.8529      | 0.8627      | 0.8627      | 0.8554      | 0.8578      | 0.8627      | 0.8578      | 0.8578      | 0.8652      | 0.8652      | 0.8603      | 0.8603      | 0.8603      | 0.8603      | 0.8578      |
| 0.1226        | 48.0  | 16032 | 0.7035          | 0.8578   | 0.8993 | 0.8786         | 0.8554      | 0.8554      | 0.8554      | 0.8529      | 0.8480      | 0.8603      | 0.8652      | 0.8627      | 0.8554      | 0.8529      | 0.8603      | 0.8578      | 0.8603      | 0.8529      | 0.8603      | 0.8603      |
| 0.1178        | 49.0  | 16366 | 0.7514          | 0.8480   | 0.8935 | 0.8708         | 0.8554      | 0.8529      | 0.8578      | 0.8505      | 0.8529      | 0.8578      | 0.8603      | 0.8554      | 0.8529      | 0.8603      | 0.8627      | 0.8554      | 0.8505      | 0.8529      | 0.8456      | 0.8480      |
| 0.1519        | 50.0  | 16700 | 0.7561          | 0.8554   | 0.8985 | 0.8769         | 0.8603      | 0.8652      | 0.8603      | 0.8554      | 0.8529      | 0.8578      | 0.8603      | 0.8578      | 0.8627      | 0.8627      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8578      | 0.8578      |
| 0.1126        | 51.0  | 17034 | 0.8944          | 0.8505   | 0.8961 | 0.8733         | 0.8505      | 0.8529      | 0.8529      | 0.8554      | 0.8529      | 0.8578      | 0.8578      | 0.8578      | 0.8505      | 0.8529      | 0.8505      | 0.8505      | 0.8505      | 0.8505      | 0.8529      | 0.8529      |
| 0.0836        | 52.0  | 17368 | 0.7771          | 0.8554   | 0.8981 | 0.8767         | 0.8554      | 0.8578      | 0.8627      | 0.8578      | 0.8554      | 0.8652      | 0.8652      | 0.8652      | 0.8627      | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8603      | 0.8578      | 0.8578      |
| 0.0548        | 53.0  | 17702 | 0.8406          | 0.8603   | 0.9026 | 0.8814         | 0.8578      | 0.8603      | 0.8603      | 0.8603      | 0.8529      | 0.8603      | 0.8603      | 0.8578      | 0.8554      | 0.8578      | 0.8603      | 0.8578      | 0.8578      | 0.8578      | 0.8578      | 0.8603      |
| 0.1526        | 54.0  | 18036 | 0.7625          | 0.8627   | 0.9038 | 0.8833         | 0.8554      | 0.8627      | 0.8603      | 0.8603      | 0.8603      | 0.8652      | 0.8652      | 0.8652      | 0.8652      | 0.8676      | 0.8676      | 0.8676      | 0.8676      | 0.8676      | 0.8701      | 0.8652      |
| 0.0541        | 55.0  | 18370 | 0.7941          | 0.8603   | 0.9009 | 0.8806         | 0.8627      | 0.8652      | 0.8627      | 0.8627      | 0.8627      | 0.8676      | 0.8652      | 0.8652      | 0.8652      | 0.8652      | 0.8676      | 0.8627      | 0.8627      | 0.8627      | 0.8603      | 0.8603      |
| 0.1095        | 56.0  | 18704 | 0.7427          | 0.8578   | 0.8997 | 0.8787         | 0.8529      | 0.8652      | 0.8627      | 0.8627      | 0.8603      | 0.8701      | 0.8652      | 0.8652      | 0.8603      | 0.8676      | 0.8676      | 0.8652      | 0.8603      | 0.8603      | 0.8603      | 0.8578      |
| 0.083         | 57.0  | 19038 | 0.8671          | 0.8627   | 0.9048 | 0.8838         | 0.8603      | 0.8603      | 0.8603      | 0.8578      | 0.8652      | 0.8627      | 0.8652      | 0.8627      | 0.8578      | 0.8627      | 0.8627      | 0.8627      | 0.8554      | 0.8603      | 0.8603      | 0.8627      |
| 0.0343        | 58.0  | 19372 | 0.9389          | 0.8529   | 0.8969 | 0.8749         | 0.8578      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8578      | 0.8652      | 0.8578      | 0.8529      | 0.8627      | 0.8603      | 0.8578      | 0.8529      | 0.8505      | 0.8505      | 0.8529      |
| 0.0339        | 59.0  | 19706 | 0.8996          | 0.8505   | 0.8946 | 0.8726         | 0.8529      | 0.8529      | 0.8529      | 0.8554      | 0.8480      | 0.8578      | 0.8578      | 0.8529      | 0.8529      | 0.8505      | 0.8578      | 0.8505      | 0.8480      | 0.8529      | 0.8529      | 0.8505      |
| 0.0598        | 60.0  | 20040 | 0.8919          | 0.8554   | 0.8985 | 0.8769         | 0.8554      | 0.8578      | 0.8554      | 0.8529      | 0.8529      | 0.8603      | 0.8603      | 0.8578      | 0.8529      | 0.8603      | 0.8652      | 0.8627      | 0.8554      | 0.8529      | 0.8505      | 0.8529      |
| 0.0834        | 61.0  | 20374 | 0.8742          | 0.8554   | 0.8981 | 0.8767         | 0.8529      | 0.8554      | 0.8554      | 0.8529      | 0.8554      | 0.8603      | 0.8603      | 0.8603      | 0.8554      | 0.8603      | 0.8603      | 0.8578      | 0.8578      | 0.8578      | 0.8554      | 0.8554      |
| 0.1174        | 62.0  | 20708 | 0.8676          | 0.8554   | 0.8985 | 0.8769         | 0.8505      | 0.8505      | 0.8554      | 0.8554      | 0.8554      | 0.8603      | 0.8603      | 0.8578      | 0.8627      | 0.8603      | 0.8627      | 0.8627      | 0.8554      | 0.8578      | 0.8554      | 0.8529      |
| 0.1133        | 63.0  | 21042 | 0.8956          | 0.8505   | 0.8950 | 0.8727         | 0.8529      | 0.8505      | 0.8505      | 0.8505      | 0.8529      | 0.8529      | 0.8578      | 0.8554      | 0.8529      | 0.8529      | 0.8529      | 0.8554      | 0.8578      | 0.8529      | 0.8529      | 0.8505      |
| 0.0481        | 64.0  | 21376 | 0.8914          | 0.8529   | 0.8969 | 0.8749         | 0.8578      | 0.8529      | 0.8529      | 0.8529      | 0.8529      | 0.8529      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8603      | 0.8578      | 0.8554      | 0.8578      | 0.8554      | 0.8529      |
| 0.045         | 65.0  | 21710 | 0.8945          | 0.8529   | 0.8966 | 0.8747         | 0.8554      | 0.8529      | 0.8529      | 0.8480      | 0.8505      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8529      | 0.8578      | 0.8578      | 0.8578      | 0.8603      | 0.8554      | 0.8529      |
| 0.0399        | 66.0  | 22044 | 0.9033          | 0.8554   | 0.8985 | 0.8769         | 0.8554      | 0.8505      | 0.8505      | 0.8529      | 0.8505      | 0.8554      | 0.8554      | 0.8529      | 0.8529      | 0.8554      | 0.8603      | 0.8554      | 0.8578      | 0.8578      | 0.8554      | 0.8554      |
| 0.0459        | 67.0  | 22378 | 0.8928          | 0.8529   | 0.8966 | 0.8747         | 0.8529      | 0.8505      | 0.8480      | 0.8529      | 0.8529      | 0.8554      | 0.8578      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8554      | 0.8529      |
| 0.066         | 68.0  | 22712 | 0.9165          | 0.8554   | 0.8985 | 0.8769         | 0.8578      | 0.8505      | 0.8529      | 0.8529      | 0.8505      | 0.8554      | 0.8554      | 0.8578      | 0.8529      | 0.8529      | 0.8603      | 0.8554      | 0.8554      | 0.8578      | 0.8578      | 0.8554      |
| 0.0278        | 69.0  | 23046 | 0.9054          | 0.8529   | 0.8966 | 0.8747         | 0.8505      | 0.8505      | 0.8505      | 0.8554      | 0.8480      | 0.8554      | 0.8554      | 0.8554      | 0.8529      | 0.8529      | 0.8554      | 0.8529      | 0.8529      | 0.8554      | 0.8554      | 0.8529      |
| 0.0753        | 70.0  | 23380 | 0.9116          | 0.8554   | 0.8985 | 0.8769         | 0.8529      | 0.8505      | 0.8529      | 0.8505      | 0.8505      | 0.8554      | 0.8554      | 0.8578      | 0.8554      | 0.8529      | 0.8578      | 0.8529      | 0.8554      | 0.8554      | 0.8554      | 0.8554      |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.2.1+cu121
- Datasets 2.16.1
- Tokenizers 0.19.1