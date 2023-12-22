### A simple neural network using python
This Python code encompasses a modular design for constructing a simple neural network. It comprises various layer types, including fully connected, activation, convolutional, and flatten layers, integrated within a network class. Each layer utilizes both forward and backward propagation methods, along with activation functions like tanh and a loss function based on mean squared error.

The code further illustrates the application of neural networks for both the XOR problem and the MNIST dataset. Various examples are presented, including a fully connected neural network, a convolutional neural network (CNN), and a CNN with convolutional layers specifically tailored for the MNIST dataset.

### **Training a hybrid quantum-classical neural network for binary classification**
The quantum circuit is used to extract non-linear features from the input data, and the classical neural network is used to combine these features and make predictions. The code demonstrates how quantum circuits can be integrated with classical neural networks to improve the performance of machine learning models on complex tasks.

### Training a Convolutional Neural Network (CNN) using PyTorch on the CIFAR-10 dataset
This code illustrates the process of building a simple CNN from scratch for image classification using the CIFAR-10 dataset. We further showcase the fine-tuning of a pre-trained ResNet-18 model on a custom flower classification dataset. Fine-tuning is a powerful technique that uses the knowledge gained from training on a large dataset (such as ImageNet) to increase the performance on a smaller, domain-specific dataset.

### A quantum convolutional neural network (QCNN) model for image classification
This code implements a quanvolutional neural network (QCNN) for image classification using the PennyLane framework and Tensorflow. A QCNN is a type of neural network that uses quantum circuits to extract features from images. The code first loads the MNIST dataset, which contains images of handwritten digits. Then, it defines a quantum circuit that can be used to convolve images. The circuit is a variational circuit, which means that its parameters can be adjusted during training. Finally, the code trains a QCNN model using the quantum preprocessed images and compares its performance to a classical CNN model that is trained on the raw images.
