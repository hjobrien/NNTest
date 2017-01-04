from MNIST_src import mnist_loader
from MNIST_src import network
import matplotlib.pyplot as plt


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net1 = network.Network([784, 30, 30, 10], weight_init=0)
net2 = network.Network([784, 20, 20, 10], weight_init=0)
net3 = network.Network([784, 10, 10, 10], weight_init=0)


end_percent = 0.005
learning_rates = [1 / 2]


net1_result = net1.SGD(training_data=training_data, epochs=20, learning_batch_size=10, learning_schedule=learning_rates,
                       evaluation_data=validation_data, termination_accuracy=end_percent, termination_duration=3,
                       test_data=test_data)

net2_result = net2.SGD(training_data=training_data, epochs=20, learning_batch_size=10, learning_schedule=learning_rates,
                       evaluation_data=validation_data, termination_accuracy=end_percent, termination_duration=3,
                       test_data=test_data)

net3_result = net3.SGD(training_data=training_data, epochs=20, learning_batch_size=10, learning_schedule=learning_rates,
                       evaluation_data=validation_data, termination_accuracy=end_percent, termination_duration=3,
                       test_data=test_data)


random = plt.plot(net1_result[1], 'b', label='random normal initial weights') #evaluation accuracy
ordered = plt.plot(net2_result[1], 'r', label='ordered initial weights')
uni_rand = plt.plot(net3_result[1], 'g', label='uniform random initial weights')

plt.ylabel('correct classifications')
plt.xlabel('epoch')
axes = plt.gca()
axes.set_ylim(0.90, 1.0)
plt.show()

