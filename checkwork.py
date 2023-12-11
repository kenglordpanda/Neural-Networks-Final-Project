import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, num_inputs, learning_rate):
        self.weights = np.random.uniform(-0.1, 0.1, (num_inputs, 1))
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def compute_response(self, inputs):
        output_layer_in = self.bias + (self.weights.T @ inputs)
        return self.sigmoid(output_layer_in)

    def update_weights(self, inputs, target):
        delta_weights = np.zeros(self.weights.shape)
        network_response = self.compute_response(inputs)
        generated_response = np.random.rand()
        if generated_response <= network_response:
            delta_weights = (self.learning_rate * target) * inputs
        self.weights += delta_weights


def modify_reinforcement(current_reinforcement):
    modified_rates = current_reinforcement
    modified_rates.reverse()
    return modified_rates


def average_response(perceptrons, presentations):
    averages = []
    for presentation in presentations:
        sum = 0
        for perceptron in perceptrons:
            sum += perceptron.compute_response(presentation)
        averages.append(sum)
    return averages


def simulate_bandit(
    num_inputs, num_networks, num_epochs, init_reinforcement, learning_rate
):
    stimuli_average_response = [[] for _ in range(num_inputs)]
    reinforcement_rates = init_reinforcement
    perceptrons = []
    presentations = []
    for init_val in range(num_inputs):
        stimulus = []
        for index in range(num_inputs):
            if index == init_val:
                stimulus.append([1])
            else:
                stimulus.append([0])
        presentation = np.array(stimulus)
        presentations.append(presentation)
    for _ in range(num_networks):
        perceptrons.append(Perceptron(num_inputs, learning_rate))

    for epoch in range(num_epochs):
        if epoch >= num_epochs // 2:
            reinforcement_rates = modify_reinforcement(reinforcement_rates)
        for index, presentation in enumerate(presentations):
            desired_value = 0 if np.random.rand() > reinforcement_rates[index] else 1
            for perceptron in perceptrons:
                perceptron.update_weights(presentation, desired_value)
        if epoch % 20 == 0:
            averages = average_response(perceptrons, presentations)
            for i, average in enumerate(averages):
                stimuli_average_response[i].append(average)
    return stimuli_average_response


def main():
    num_stimuli = 4
    num_perceptrons = 10
    num_epochs = 600
    reinforcement = [0.2, 0.4, 0.6, 0.8]
    learning_rate = 0.1
    values = simulate_bandit(
        num_stimuli, num_perceptrons, num_epochs, reinforcement, learning_rate
    )
    print(f"Bandit Simulated")


if __name__ == "__main__":
    main()
