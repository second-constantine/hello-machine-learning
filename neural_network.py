import math
import random

LEARNING_RATE = 0.5


def squash(total_net_input):
    return 1 / (1 + math.exp(-total_net_input))


class Neuron:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.output = 0

    def calculate_inputs(self):
        total = 0
        for (i, input) in enumerate(self.inputs):
            total += input * self.weights[i]
        return total

    def calculate_error(self, target_output):
        return 0.5 * math.pow(target_output - self.output, 2)

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = squash(self.calculate_inputs())
        return self.output

    def diff_output(self, target_output):
        return -(target_output - self.output)

    def calculate(self):
        return self.output * (1 - self.output)

    def calculate_error_input(self, target_output):
        return self.diff_output(target_output) * self.calculate()


class NeuronLayer:
    def __init__(self, neuron_amount, bias):
        self.bias = bias
        self.neurons = []
        for i in range(neuron_amount):
            self.neurons.append(Neuron())

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.hidden_layer = NeuronLayer(num_hidden, random.uniform(0, 1))
        self.output_layer = NeuronLayer(num_outputs, random.uniform(0, 1))
        for neuron in self.output_layer.neurons:
            for (i, v) in enumerate(self.hidden_layer.neurons):
                neuron.weights.append(random.uniform(0, 1))
        for neuron in self.hidden_layer.neurons:
            for i in range(num_inputs):
                neuron.weights.append(random.uniform(0, 1))

    def feed_forward(self, inputs):
        return self.output_layer.feed_forward(self.hidden_layer.feed_forward(inputs))

    def predication(self, inputs):
        return self.feed_forward(inputs)[0]

    def print_prediction(self, inputs):
        print ("%s -> %.2f%%" % (str(inputs), self.predication(inputs) * 100))

    def train(self, inputs, outputs):
        self.feed_forward(inputs)
        errors_input_for_output_layer = []
        output_layer_neurons = self.output_layer.neurons
        for (i, output_neuron) in enumerate(output_layer_neurons):
            errors_input_for_output_layer.append(output_neuron.calculate_error_input(outputs[i]))
        errors_input_for_hidden_layer = []
        hidden_layer_neurons = self.hidden_layer.neurons
        for (i, hidden_neuron) in enumerate(hidden_layer_neurons):
            error = 0
            for (j, output_neuron) in enumerate(output_layer_neurons):
                error += errors_input_for_output_layer[j] * output_neuron.weights[i]
            errors_input_for_hidden_layer.append(error * hidden_neuron.calculate())
        for (i, neuron) in enumerate(output_layer_neurons):
            for (w, weight) in enumerate(neuron.weights):
                weight -= LEARNING_RATE * errors_input_for_output_layer[i] * neuron.inputs[w]
                neuron.weights[w] = weight
        for (i, neuron) in enumerate(hidden_layer_neurons):
            for (w, weight) in enumerate(neuron.weights):
                weight -= LEARNING_RATE * errors_input_for_hidden_layer[i] * neuron.inputs[w]
                neuron.weights[w] = weight

    def calculate_error(self, value):
        total_error = 0
        for (a, v) in enumerate(value):
            inputs = value[a][0]
            outputs = value[a][1]
            self.feed_forward(inputs)
            for (i, output) in enumerate(outputs):
                total_error += self.output_layer.neurons[i].calculate_error(output)
        return total_error


if __name__ == "__main__":
    training_set = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    nn = NeuralNetwork(len(training_set[0][0]), 10, len(training_set[0][1]))
    for i in range(100000):
        choice = random.randint(0, len(training_set) - 1)
        nn.train(training_set[choice][0], training_set[choice][1])

    nn.print_prediction([0, 0])
    nn.print_prediction([1, 0])
    nn.print_prediction([0, 1])
    nn.print_prediction([1, 1])
    nn.print_prediction([1, 2])
    nn.print_prediction([2, 2])
    nn.print_prediction([2, 1])
    print (nn.predication([2, 0]))
