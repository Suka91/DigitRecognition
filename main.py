import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 50, 30,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)