import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.fit(training_data, validation_data, epochs = 1, alpha = 1.0, batch_size = 10)