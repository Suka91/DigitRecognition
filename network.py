import numpy as np

class Network:
	def __init__(self, layers):
		
		self.num_layers = len(layers)
		self.weights = np.array([ np.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])])
		self.bias = np.array([ np.random.randn(x,1) for x in layers[1:]])

	def fit(self, train_data, validation_data, epochs = 1, alpha = 0.1, test_data = None, batch_size = None):

		print("Epoch: -1 of",epochs,"- Score: ",self.evaluate(validation_data)," / ",len(validation_data))
		for i in range(epochs):
			if batch_size is None:
				self.update_network(train_data, alpha)
			else:
				batches = np.split(train_data, len(train_data)/batch_size)
				for batch in batches:
					self.update_network(batch, alpha)
			print("Epoch: ",i," of",epochs,"- Score: ",self.evaluate(validation_data)," / ",len(validation_data))

	def update_network(self, train_data, alpha):

		weights_grad_total = [0 for x in range(self.num_layers-1)]
		bias_grad_total = [0 for x in range(self.num_layers-1)]
		for x,y in train_data:
			bias_grad, weights_grad = self.backprop(x, y)
			bias_grad_total = [ bgt + bg for bgt,bg in zip(bias_grad_total, bias_grad)]
			weights_grad_total = [ wgt + wg for wgt,wg in zip(weights_grad_total, weights_grad)]
		self.weights = [w - alpha/len(train_data)*wgt for w,wgt in zip(self.weights, weights_grad_total)]
		self.bias = [b - alpha/len(train_data)*bgt for b,bgt in zip(self.bias, bias_grad_total)]

	def backprop(self, x, y):

		zs = []
		activations = [x]
		weights_grad = [0 for x in range(self.num_layers-1)]
		bias_grad = [0 for x in range(self.num_layers-1)]
		for i in range(self.num_layers-1):
			z = np.dot(self.weights[i],activations[i]) + self.bias[i]
			zs.append(z)
			activations.append(sigmoid(z))

		cost_gradient_respect_a = self.cost_gradient_respect_a(activations[-1], y)
		delta = cost_gradient_respect_a
		bias_grad[-1] =  cost_gradient_respect_a
		weights_grad[-1] =  np.dot(cost_gradient_respect_a,activations[-2].T)
		for i in range(1,self.num_layers-1):
			delta = np.dot(self.weights[-i].T, delta) * sigmoid_prime(zs[-i-1])
			bias_grad[-i-1] = delta
			weights_grad[-i-1] = np.dot(delta, activations[-i-2].T)
		return bias_grad, weights_grad
		
	def evaluate(self,validation_data):
		score = 0
		for x,y in validation_data:
			activations = x
			for i in range(self.num_layers-1):
				activations = sigmoid(np.dot(self.weights[i], activations)+self.bias[i])
			output = np.argmax(activations)
			if(isinstance(y, (np.ndarray))):
				y = np.argmax(y)
			if(output == y):
				score+=1
		return score

	def cost_gradient_respect_a(self, a, y):
		return a-y

def sigmoid(x):
	return 1/(1+np.exp(-x))
def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))