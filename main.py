import mnist_loader
import network
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
'''
net = network.Network([784, 30, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10,lmbda = 0.001)
net.saveHistory("Network1","784_30_10 epochs = 30, alpha = 3.0, batch_size = 10")


net = network.Network([784, 30, 10] , reduce_weights_deviation = False)
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10)
net.saveHistory("Network2","784_30_10 reduce_weights_deviation = False epochs = 30, alpha = 1.0, batch_size = 10")

net = network.Network([784, 30, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10)
net.saveHistory("Network3","784_30_10 epochs = 30, alpha = 1.0, batch_size = 10")

net = network.Network([784, 50, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10)
net.saveHistory("Network4","784_50_10 epochs = 30, alpha = 1.0, batch_size = 10")

net = network.Network([784, 50, 30, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10)
net.saveHistory("Network5","784_50_30_10 epochs = 30, alpha = 1.0, batch_size = 10")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10)
net.saveHistory("Network6","784_100_10 epochs = 30, alpha = 1.0, batch_size = 10")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10,lmbda=0.01)
net.saveHistory("Network7","784_100_10 epochs = 30, alpha = 1.0, batch_size = 10 lmbda=0.01")

net = network.Network([784, 150, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10)
net.saveHistory("Network8","784_150_10 epochs = 30, alpha = 1.0, batch_size = 10")

net = network.Network([784, 90, 40, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 60, alpha = 1.0, batch_size = 10)
net.saveHistory("Network9","784_90_40_10 epochs = 60, alpha = 1.0, batch_size = 10")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10,lmbda = 0.001)
net.saveHistory("Network10","784_100_10 epochs = 30, alpha = 1.0, batch_size = 10 lmbda=0.001")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10,lmbda = 0.0001)
net.saveHistory("Network11","784_100_10 epochs = 30, alpha = 1.0, batch_size = 10 lmbda=0.0001")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10,lmbda = 0.0005)
net.saveHistory("Network12","784_100_10 epochs = 30, alpha = 1.0, batch_size = 10 lmbda=0.0005")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 1.0, batch_size = 10,lmbda = 0.1)
net.saveHistory("Network13","784_100_10 epochs = 30, alpha = 1.0, batch_size = 10 lmbda=0.1")

net = network.Network([784, 150, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 50, alpha = 1.0, batch_size = 10)
net.saveHistory("Network14","784_150_10 epochs = 50, alpha = 1.0, batch_size = 10")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 0.66, batch_size = 10)
net.saveHistory("Network15","784_100_10 epochs = 30, alpha = 0.66, batch_size = 10")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 0.33, batch_size = 10)
net.saveHistory("Network16","784_100_10 epochs = 30, alpha = 0.33, batch_size = 10")

net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 30, alpha = 0.1, batch_size = 10)
net.saveHistory("Network17","784_100_10 epochs = 30, alpha = 0.1, batch_size = 10")


net = network.Network([784, 100, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 60, alpha = 0.66, batch_size = 10, lmbda=0.00001)
net.saveHistory("Network18","784_100_10 epochs = 60, alpha = 0.66, batch_size = 10, lmbda = 0.00001")

net = network.Network([784, 150, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 40, alpha = 0.66, batch_size = 10)
net.saveHistory("Network19","784_150_10 epochs = 40, alpha = 0.66, batch_size = 10")

net = network.Network([784, 150, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 40, alpha = 0.75, batch_size = 10)
net.saveHistory("Network20","784_150_10 epochs = 40, alpha = 0.75, batch_size = 10")

net = network.Network([784, 150, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 40, alpha = 0.66, batch_size = 10,lmbda=0.00001)
net.saveHistory("Network21","784_150_10 epochs = 40, alpha = 0.66, batch_size = 10,lmbda=0.00001")
'''
net = network.Network([784, 200, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 40, alpha = 0.75, batch_size = 10)
net.saveHistory("Network22","784_200_10 epochs = 40, alpha = 0.75, batch_size = 10")

net = network.Network([784, 250, 10])
net.fit(training_data, validation_data,history_validation=True,history_training=True, epochs = 40, alpha = 0.75, batch_size = 10)
net.saveHistory("Network23","784_250_10 epochs = 40, alpha = 0.75, batch_size = 10")