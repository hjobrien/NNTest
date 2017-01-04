import mnist_loader
import networkExample

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = networkExample.Network([784, 30, 10], cost=networkExample.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
