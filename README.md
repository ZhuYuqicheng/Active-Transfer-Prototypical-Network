# Active Transfer Prototypical Network (ATPN)

Implementing state-of-the-art machine learning algorithms is challenging for conventional industrial companies due to the enormous demand for a labeled training dataset. However, labeling a significant amount of time-series data is time-consuming or impossible even for domain experts. Therefore, a reliable algorithm that can significantly reduces the labeling effort is a crucial element in unlocking the potential of artificial intelligence in more domains.

In this repository, a novel active learning framework based on a prototypical network was implemented (built on top of [modAL: A modular active learning framework](https://modal-python.readthedocs.io/en/latest/) ). Standard **Active Learning** reduces labeling effort by actively querying informative instances during training process, but it strongly depends on the performance of the initial model. On the other hand, **Prototypical Network** leverages the prior knowledge of a pre-trained encoder to train the model with a very small dataset. However, the support set of prototypical network should be representative. The combination of the both methods has potential to compensate the disadvantages from both sides. This model was initially proposed for time-series data, but can be easily generalized to other data format.

## Algorithm Structure

<p align="center">
  <img src=pictures/structure.png alt="Sublime's custom image"/>
</p>

The model is trained iteratively like standard active learning. But the initial model is a prototypical network with a pre-trained encoder, which is trained on labeled dataset of similar tasks. The queried instances will be used to update the support set of prototypical network, in order to get representative prototypes step by step. They are also used to fine-tuned the last layer of the pre-trained encoder, so that the pre-trained encoder can quickly adapt to current dataset even it was trained on quite different dataset.

## Quick Start

