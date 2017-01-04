import json
import random
import sys
import numpy as np
import MNIST_src.Cost_Functions as costs


class Cross_Entropy_Cost:


    #the actual cost function
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

    """how to compute the output error for this cost function
    first param is included to support other cost functions with different delta methods"""

    @staticmethod
    def delta(_, a, y):
        return a-y

class Quadratic_Cost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class Network():

    """
    initialize a neural network class
    sizes is a matrix of the number of neurons in each layer (and hence also the number of layers in the network
    """
    def __init__(self, sizes, cost=Cross_Entropy_Cost, weight_init = 0):
        self.num_layers = len(sizes)
        self.sizes = sizes

        if(weight_init == 0):
            self.biases, self.weights = self.ordered_weight_initializer()
        elif (weight_init == 1):
            self.biases, self.weights = self.default_weight_initializer()
        else:
            self.biases, self.weights = self.uniform_random_weight_initializer()
        self.cost = cost

    # initializes the weights of the neural network to a random standard normal distribution
    # assumes first layer is input layer so no biases are generated for the neurons in the first layer
    def default_weight_initializer(self):
        biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return (biases, weights)


    def ordered_weight_initializer(self):
        biases = [np.linspace(-1, 1, num=y).reshape(y,1) for y in self.sizes[1:]]
        weights = [np.vstack(tuple([np.linspace(-1/np.sqrt(x), 1/np.sqrt(x), num=x) for _ in range(y)])) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
        return (biases, weights)


    def uniform_random_weight_initializer(self):
        biases = [np.random.uniform(-1, -1, (y, 1)) for y in self.sizes[1:]]
        weights = [np.random.uniform(-1, 1, (y, x)) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return (biases, weights)

    """initializes weights of neural network to a random distribution with a mean = 0 and a standard deviation = 1"""
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    """
    returns the output of the network for some set of inputs 'input'
    """
    def feedforward(self, input):
        for b, w in zip(self.biases, self.weights):
            input = sigmoid(np.dot(w, input) + b)
        return input

    """
    implements stochastic gradient descent on this neural network
    training_data is a list of tuples (x, y) representing the training inputs and their expected outputs
    if test data is provided, network will print it progress relative to the test data each epoch, this has a time cost
    lmbda is the regularization parameter for L2 regularization
    """
    def SGD(self, training_data, epochs, learning_batch_size, learning_schedule, evaluation_data, termination_accuracy,
            termination_duration, test_data=None, print_eval_cost=False, print_eval_accur=True, print_train_cost=False,
            print_train_accur=False, lmbda=0):

        results = []
        if(test_data):
            n_test = len(test_data)
        n = len(training_data)

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        historical_accuracy = [None]*termination_duration
        # for i in range(epochs):
        epoch_number = 0
        for learning_rate in learning_schedule:
            print('\n learning rate is now: {0}'.format(learning_rate))

            # uncomment this to add early termination (to combat overfitting)
            # while (not self.should_stop(historical_accuracy, termination_accuracy)):
            for i in range(epochs):

                random.shuffle(training_data)
                learning_batches = [training_data[k:(k+learning_batch_size)] for k in range(0, n, learning_batch_size)]
                # TODO: change loop to single matrix operation (need to concat. vectors)
                for learning_batch in learning_batches:
                    self.update_learning_batch(learning_batch, learning_rate, lmbda, len(training_data))

                print("Epoch {0} training complete".format(epoch_number))

                if(print_train_cost):
                    cost = self.total_cost(training_data, lmbda)
                    training_cost.append(cost)
                    print('training cost: {0}'.format(cost))
                if(print_train_accur):
                    accuracy = self.accuracy(training_data, convert=True)
                    training_accuracy.append(accuracy)
                    print('training accuracy: {0} / {1}'.format(accuracy*len(training_data), n))
                if(print_eval_cost):
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                    evaluation_cost.append(cost)
                    print('evaluation cost: {0}'.format(cost))

                accuracy = self.accuracy(evaluation_data)
                if (print_eval_accur):
                    evaluation_accuracy.append(accuracy)
                    print('evaluation accuracy: {0}'.format(accuracy * len(evaluation_data)))
                historical_accuracy.pop(0)
                historical_accuracy.append(accuracy)
                epoch_number += 1

                print()
            historical_accuracy = [None]*termination_duration
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    """
    returns true if all the elements of the historical accuracies were below a threshold"""
    def should_stop(self, historical_accuracy, threshold):
        print(historical_accuracy)
        return all(accuracy is not None for accuracy in historical_accuracy)\
               and all(diff < threshold for diff in (abs(np.diff(historical_accuracy))))
        # return all((accuracy is not None) and (accuracy > threshold) for accuracy in historical_accuracy)
    """
    updates the neural networks weights based on a batch of samples and a learning rate by applying
    the gradient descent using backpropagation over a single batch of data"""
    def update_learning_batch(self, learning_batch, learning_rate, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in learning_batch:

            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1-learning_rate*(lmbda/n)) * w -(learning_rate / len(learning_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate / len(learning_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    """
    returns a tuple of (nabla_b, nabla_w) representing the gradient of the the cost function
    nabla_b and nabla_w are lists of numpy arrays of the gradient at each particular layer """
    def backpropagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x]   #list of activations, layer by layer
        zs = [] # list of z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l here refers to layers from the back forward, so the last layer has L = 1, the second to last has L = 2
        for L in range(2, self.num_layers):
            z = zs[-L]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-L+1].transpose(), delta) * sp
            nabla_b[-L] = delta
            nabla_w[-L] = np.dot(delta, activations[-L-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if(convert):
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results) / len(results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0
        for x, y in data:
            a = self.feedforward(x)
            if(convert):
                y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {'sizes': self.sizes, 'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases], 'cost': str(self.cost.__name__)}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()

    """
    returns the number of inputs for where the neural network outputs the correct output.
    output is determined by choosing the output neuron with the largest value (highest activation)
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def load(filename):
    f = open(filename)
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(data['sizes'], cost=cost)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net



def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1
    return e

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1- sigmoid(z))

