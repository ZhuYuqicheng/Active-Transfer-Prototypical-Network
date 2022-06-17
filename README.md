# Active Transfer Prototypical Network (ATPN)

> **Author**: **Yuqicheng Zhu**
> 
> **Institute**: TUM Department of Electrical and Computer Engineering
> **Supervisors**:
> - Prof. Dr.-Ing. **Klaus Diepold**
> - M. Sc. **Mohamed Ali Tnani**
> - Dipl. -Ing. **Timo Jahnz**

Implementing state-of-the-art machine learning algorithms is challenging for conventional industrial companies due to the enormous demand for a labeled training dataset. However, labeling a significant amount of time-series data is time-consuming or impossible even for domain experts. Therefore, a reliable algorithm that can significantly reduce the labeling effort is crucial in unlocking the potential of artificial intelligence in more domains.

In this repository, a novel active learning framework based on a prototypical network was implemented (built on top of [modAL: A modular active learning framework](https://modal-python.readthedocs.io/en/latest/) ). Standard **Active Learning** reduces labeling effort by actively querying informative instances during the training process, but it strongly depends on the performance of the initial model. On the other hand, **Prototypical Network** leverages the prior knowledge of a pre-trained encoder to train the model with a very small dataset. However, the support set of the prototypical network should be representative. Therefore, the combination of both methods has the potential to compensate for the disadvantages on both sides. This model was initially proposed for time-series data but can be easily generalized to other data formats.

## Algorithm Structure

<p align="center">
  <img src=pictures/structure.png alt="Sublime's custom image"/>
</p>

The model is trained iteratively like standard active learning. But the initial model is a prototypical network with a pre-trained encoder trained on a labeled dataset of similar tasks. The queried instances will be used to update the support set of the prototypical network to get more and more representative prototypes. They are also used to fine-tune the last layer of the pre-trained encoder so that the pre-trained encoder can quickly adapt to the current dataset even if it was trained on a quite different dataset.

## Active Transfer Prototypical Network Tutorial

You can check out the tutorial in Colab and try your own experiment settings here: [Active Transfer Prototypical Network Tutorial](https://colab.research.google.com/drive/1SrCqQ7kzmZDGTTZ22cEdZqL-OGJYJcyt?usp=sharing).

