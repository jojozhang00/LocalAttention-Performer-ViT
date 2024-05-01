# Transformers as visual encoders

The goal of this project is to benchmark different Performer-architectures as well as localattention Transformer-architectures within the Vision Transformer framework across three image classification datasets. Local Attention models aim to capture more localized features within an image, while Performer models seek to optimize computational efficiency by approximating the softmax function in the attention mechanism. This research identifies key trade-offs between computational speed and task performance. Performer-Softmax ViT shows an ideal trade-off.

Experiments are using the same modified ViT models. The difference is the training and testing process on three tasks. The process for the MNIST dataset can be simply replaced by experiment_STL10.py and experiment_CIFAR10.py to conduct experiments on STL-10 and CIFAR-10.
