---
layout: post
title: "Knowledge Distillation — A survey"
date: 2022-02-13
---

This blog is mostly my learnings from the paper: Knowledge Distillation: A Survey and I try to present my summary which should hopefully be easier to follow.

## **Problem**

Deep learning models being huge with billions of parameters, are very difficult to deploy to devices with limited resources like phones & embedded devices or to be used for real-time inferencing or serving where the typical latency requirements are in milliseconds(≤500 ms).

## **Solution**

* One of the popular model compression and acceleration techniques is the knowledge distillation(KD) technique, where we transfer knowledge from a large model to a small model.

* KD system has 3 key components: *knowledge*, *distillation algorithm*, *teacher-student architecture*

### Knowledge

Different kinds of knowledge can be

* **Response-based Knowledge**: this knowledge mimics the final prediction of the teacher model.

![Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)](https://cdn-images-1.medium.com/max/2000/1*1p7H2hSUfuCHAu0_4l3Q7g.png)*Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)*

* **Feature-based knowledge**: It’s well known how deep neural networks use various layers for various kinds of feature representation. On the same note, both the output of the last layer(response-based knowledge) and the output of intermediate layers are used as knowledge to supervise the training of the student model.

![Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)](https://cdn-images-1.medium.com/max/2000/1*FFHR3coJJz_9vfpZX9PKHw.png)*Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)*

* **Relation-based knowledge (also called Shared representation knowledge)**: this knowledge explores the relationship between intermediate layers, for example using inner product between features from 2 different layers as knowledge. More concretely, teacher’s knowledge about the relation between different input examples is transferred to the student model:

![Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)](https://cdn-images-1.medium.com/max/2000/1*5vDdv1UEVsOQSRKAxqXklw.png)*Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)*

![Sources of response-based knowledge, feature-based knowledge & relation-based knowledge in a deep teacher network | Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)](https://cdn-images-1.medium.com/max/2000/1*7K6cX6QstfYXh_YLFvmGBw.png)*Sources of response-based knowledge, feature-based knowledge & relation-based knowledge in a deep teacher network | Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)*

### Distillation schemes

3 main categories depending upon whether teacher and student models are simultaneously updated or not:

* **Offline distillation**: it’s the easiest to implement. A teacher model is first pre-trained on a training dataset and knowledge from the teacher model is transferred to train the student model.

* **Online distillation**: Teacher and student models are updated simultaneously in the same training process. We can parallelize the training here by using several distribution strategies(data and/or model), making this process efficient.

* **Self-distillation**: teacher and student models are of the same size and same architecture.These can either be a single model representing teacher as well as student or we may have two instances of the same model: one being teacher and another being student. The main idea here is that knowledge from deeper layers can be used to train the shallow layers. The process usually makes the student model more robust and accurate.

![Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)](https://cdn-images-1.medium.com/max/2000/1*maSLJNT6o72zf9v8jzl_7A.png)*Figure adapted from Jianping Gou et al. (2020) | 📝 [Paper](https://arxiv.org/abs/2006.05525)*

## Teacher-student architecture:

The most common student architectures are:

* a quantized version of the teacher model

* a smaller version of the teacher model with few layers and fewer neurons per layer

* same model as the teacher

## Distillation algorithms

* **Adversarial distillation: **Uses the concept of Generative adversarial networks(GANs) where a generator model poses several difficult questions to the teacher model and the student model learns how to answer those questions from the teacher.

* **Multi-teacher distillation**: multiple teacher models are used to provide distinct kinds of knowledge to the student model

* **Cross-modal(cross-disciplinary) distillation: **The teacher model which is trained in one modality(say vision domain) is used to train a student model from a different modality(say text-domain). Example application: visual question-answering

* **Graph-based distillation[²](https://arxiv.org/abs/1907.02226):** Knowledge of the embedding procedure of the teacher network is distilled into a graph, which is in turn used to train the student model

* **Attention-based distillation**: here attention maps are used to transfer knowledge about feature embeddings to the student model

* **Data-free distillation**: in absence of a training dataset(due to privacy, security, etc. issues), synthetic data is generated from the teacher model or GANs are used.

* **Quantized distillation**: transfer knowledge from a high-precision teacher(say 32-bit) to a low-precision student model(say 8-bit)

* **Lifelong distillation: **Continuously learned knowledge of the teacher model is transferred to the student model ([Youtube video reference](https://www.youtube.com/watch?v=t3Ee5fA8mCo&ab_channel=Yung-SungChuang))

* **Neural architecture search(NAS)-based distillation: **Use AutoML to automatically identify the appropriate student model in terms of deciding the apt capacity gap between teacher and student models.

## Performance aspects and conclusions

* offline distillation does feature-based knowledge transfer

* online distillation does response-based knowledge transfer

* the performance of student models can be improved by knowledge transfer from teacher(high-capacity) models

## Applications of KD

* KD has found applications in visual recognition, NLP, speech recognition, various other applications

## Challenges

* It’s a challenge to measure the quality of knowledge or quality of student-teacher architecture

## Future directions

* It would be useful to integrate KD with other learning schemes like reinforcement learning, adversarial learning, etc.

## References:

1. [*Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525.pdf)*

1. [*Graph-based Knowledge Distillation by Multi-head Attention Network](https://arxiv.org/abs/1907.02226)*
