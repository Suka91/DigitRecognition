import numpy as np

class Sigmoid(object):

		@staticmethod
		def fn(x):
			return 1/(1+np.exp(-x))

		@staticmethod
		def prime(x):
			return Sigmoid.fn(x)*(1-Sigmoid.fn(x))

class Relu(object):

		@staticmethod
		def fn(x):
			return x * (x > 0)

		@staticmethod
		def prime(x):
			return 1 * (x > 0)

class Network:
	def __init__(self, layers, reduce_weights_deviation = True):
		np.random.seed(42)
		self.num_layers = len(layers)
		self.bias = np.array([ np.random.randn(x,1) for x in layers[1:]])
		if(reduce_weights_deviation == True):
			self.weights = np.array([ np.random.randn(y,x)/np.sqrt(x) for x,y in zip(layers[:-1],layers[1:])])
		else:
			self.weights = np.array([ np.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])])
		self.activation_function = None

		self.history_description = ""
		self.train_accuracy = None
		self.train_cost = None
		self.train_score = None

		self.validation_accuracy = None
		self.validation_cost = None
		self.validation_score = None

	def fit(self, train_data, validation_data, 
			epochs = 1, 
			alpha = 0.1, 
			test_data = None, 
			batch_size = None, 
			lmbda = 0,
			activation = "sigmoid",
			history_validation = False,
			history_training = False):

		if(activation == "relu"):
			self.activation_function = Relu
		else:
			self.activation_function = Sigmoid

		if(history_validation == True):
			self.validation_accuracy = []
			self.validation_cost = []
			score,max_score,accuracy,cost = self.calculate_score_accuracy_cost(validation_data)
			string = "Validation: Epoch: 0 of " + str(epochs) + " - Score: " + str(score) + " / " + str(max_score) + " - Accuracy: " + str(accuracy) + " - Cost: " + str(cost)
			print(string)
			self.validation_accuracy.append(accuracy)
			self.validation_cost.append(cost)
			self.history_description += string
			self.history_description += "\n"
		if(history_training == True):
			self.train_accuracy = []
			self.train_cost = []
			score,max_score,accuracy,cost = self.calculate_score_accuracy_cost(train_data)
			string = "Training: Epoch: 0 of " + str(epochs) + " - Score: " + str(score) + " / " + str(max_score) + " - Accuracy: " + str(accuracy) + " - Cost: " + str(cost)
			print(string)
			self.train_accuracy.append(accuracy)
			self.train_cost.append(cost)
			self.history_description += string
			self.history_description += "\n"

		for i in range(epochs):
			if batch_size is None:
				self.update_network(train_data, alpha, lmbda)
			else:
				np.random.shuffle(train_data)
				batches = np.split(train_data, len(train_data)/batch_size)
				for batch in batches:
					self.update_network(batch, alpha, lmbda)
			if(history_validation == True):
				score,max_score,accuracy,cost = self.calculate_score_accuracy_cost(validation_data)
				string = "Validation: Epoch: " + str(i+1) + " of " + str(epochs) + " - Score: " + str(score) + " / " + str(max_score) + " - Accuracy: " + str(accuracy) + " - Cost: " + str(cost)
				print(string)
				self.validation_accuracy.append(accuracy)
				self.validation_cost.append(cost)
				self.history_description += string
				self.history_description += "\n"
			if(history_training == True):
				score,max_score,accuracy,cost = self.calculate_score_accuracy_cost(train_data)
				string = "Training: Epoch: " + str(i+1) + " of " + str(epochs) + " - Score: " + str(score) + " / " + str(max_score) + " - Accuracy: " + str(accuracy) + " - Cost: " + str(cost)
				print(string)
				self.train_accuracy.append(accuracy)
				self.train_cost.append(cost)
				self.history_description += string
				self.history_description += "\n"

	def update_network(self, train_data, alpha, lmbda):

		weights_grad_total = [0 for x in range(self.num_layers-1)]
		bias_grad_total = [0 for x in range(self.num_layers-1)]
		for x,y in train_data:
			bias_grad, weights_grad = self.backprop(x, y, lmbda)
			bias_grad_total = [ bgt + bg for bgt,bg in zip(bias_grad_total, bias_grad)]
			weights_grad_total = [ wgt + wg for wgt,wg in zip(weights_grad_total, weights_grad)]
		self.weights = [w - alpha/len(train_data)*wgt for w,wgt in zip(self.weights, weights_grad_total)]
		self.bias = [b - alpha/len(train_data)*bgt for b,bgt in zip(self.bias, bias_grad_total)]

	def backprop(self, x, y, lmbda):

		zs = []
		activations = [x]
		weights_grad = [0 for x in range(self.num_layers-1)]
		bias_grad = [0 for x in range(self.num_layers-1)]
		for i in range(self.num_layers-1):
			z = np.dot(self.weights[i],activations[i]) + self.bias[i]
			zs.append(z)
			activations.append(self.activation_function.fn(z))

		cost_gradient_respect_a = self.cost_gradient_respect_a(activations[-1], y)
		delta = cost_gradient_respect_a
		bias_grad[-1] =  cost_gradient_respect_a
		weights_grad[-1] =  np.dot(cost_gradient_respect_a,activations[-2].T)
		for i in range(1,self.num_layers-1):
			delta = np.dot(self.weights[-i].T, delta) * self.activation_function.prime(zs[-i-1])
			bias_grad[-i-1] = delta
			weights_grad[-i-1] = np.dot(delta, activations[-i-2].T) + 2*lmbda*self.weights[-i-1]
		return bias_grad, weights_grad

	def calculate_score_accuracy_cost(self,data):
		score = 0
		accuracy = 0
		cost = 0.0
		for x,y in data:
			a = self.feedForward(x)
			output = np.argmax(a)
			
			if(isinstance(y, (np.ndarray))):
				y = np.argmax(y)

			new_y = np.zeros((10, 1))
			new_y[y] = 1.0
			
			cost += self.cost_function(a, new_y)/len(data)
			if(output == y):
				score += 1

		accuracy = 100.0 * score / len(data) 
		return score,len(data),accuracy,cost

	def feedForward(self, x):
		activations = x
		for i in range(self.num_layers-1):
			activations = self.activation_function.fn(np.dot(self.weights[i], activations)+self.bias[i])
		return activations

	def saveHistory(self,history_name):
		file = open(history_name+"_description.txt",'w')
		file.write(self.history_description)
		file.close()

		if(self.validation_cost is not None):
			file = open(history_name+"_validation_cost.txt",'w')
			string = ""
			for record in self.validation_cost:
				string += str(record)
				string += ";"
			file.write(string)
			file.close()

		if(self.validation_accuracy is not None):
			file = open(history_name+"_validation_accuracy.txt",'w')
			string = ""
			for record in self.validation_accuracy:
				string += str(record)
				string += ";"
			file.write(string)
			file.close()

		if(self.train_cost is not None):
			file = open(history_name+"_train_cost.txt",'w')
			string = ""
			for record in self.train_cost:
				string += str(record)
				string += ";"
			file.write(string)
			file.close()

		if(self.train_accuracy is not None):
			file = open(history_name+"_train_accuracy.txt",'w')
			string = ""
			for record in self.validation_cost:
				string += str(record)
				string += ";"
			file.write(string)
			file.close()

	def cost_gradient_respect_a(self, a, y):
		return a-y

	def cost_function(self, a, y):
		return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y)*np.log(1 - a)))
