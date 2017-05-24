import random
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
from matplotlib import pyplot as plt

class CrossEntropyCost(object):
    
    @staticmethod
    def fn(a, y):
        """Quadratic cost function where a is calculated output and
        y is real output value"""
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer,
        which is derivative of cost function"""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Cross-entropy cost function where a is calculated output and
        y is real output value"""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer,
        which is derivative of cost function
        """
        return (a-y)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = np.array([np.random.randn(y, 1) for y in sizes[1:]])
        '''weights are as follows, weight[l][i][j] is weight from j-th neuron in
        l layer to i-th neuron in l+1 layer, this notation is suitable for
        numerical reasons seen later in computation.''' 
        self.weights = np.array([np.random.randn(x, y)
                        for x, y in zip(sizes[1:],sizes[:-1])])
        self.cost = cost
        self.costs = []

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, alpha,
            test_data=None,
            track_cost=True):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data is not None:
            n_test = len(test_data)
            print("Epoch -1 :", self.evaluate(test_data), " / ", n_test)
        
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)
            if track_cost is not None:
                self.costs.append(self.calculateCost(training_data))
            if test_data is not None:
                print("Epoch ", j, ":", self.evaluate(test_data), " / ", n_test)
            else:
                print("Epoch", j, "complete")


    def update_mini_batch(self, mini_batch, alpha):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``alpha``
        is the learning rate."""
        '''
        - D_b and D_w are used to store ~ gradient of the cost function
        over biases and weights measured for every training example.
        - delta_nabla_b and delta_nabla_w are used to ~ gradient of the cost
        function for single training istance,
        delta_nabla_w(l) = delta(l+1) * a.T(l)
        delta_nabla_b(l) = delta(l+1)
        delta(L) = a(L) - y
        delta(l) = (weights(l).T * delta(l+1)) .* sigmoid_prime(z(l))
        '''
        D_b = [np.zeros(b.shape) for b in self.biases]
        D_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            D_b = [db+dnb for db, dnb in zip(D_b, delta_nabla_b)]
            D_w = [dw+dnw for dw, dnw in zip(D_w, delta_nabla_w)]
        self.weights = np.array([w-(alpha/len(mini_batch))*dw
                                for w, dw in zip(self.weights, D_w)])
        self.biases = np.array([b-(alpha/len(mini_batch))*db
                                for b, db in zip(self.biases, D_b)])

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` an
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            #Apply weight and bias to input
            z = np.dot(w, activation)+b
            zs.append(z)
            #Apply activation on input
            activation = sigmoid(z)
            #Save activations (later used for calculating gradient of cost function)
            activations.append(activation)
        # backward pass
        #calculate derivative of the the cost function in output layer
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            #propagate error to current layer
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #update nablas
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def calculateCost(self, input):
        cost = 0.0
        m = len(input)
        for x,y in input:
            a = self.feedforward(x)
            cost += self.cost.fn(a,y)/m
        return cost

    def visualizeCostOverEpochs(self, cost, epochs):
        plt.plot(cost,epochs)
        plt.show()
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))