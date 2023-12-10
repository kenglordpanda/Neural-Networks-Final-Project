import matplotlib.pyplot as plt
import numpy as np

"""
Research paper does not show that bias is implemented so it is not included in this code
"""


class Perceptron:
    def __init__(self, num_inputs, learning_rate, target):
        self.inputs = np.random.rand(num_inputs)
        self.weights = np.random.uniform(-0.1, 0.1, num_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.target = target
        self.output = -1

    def compute_output(self):
        output_layer_in = self.bias + self.weights * self.inputs
        output = 0
        if output_layer_in > 0:
            output = 1
        return output

    def update_weights(self):
        delta_weights = np.zeros(self.weights.shape)
        if self.output == self.target:
            delta_weights = self.learning_rate * self.target * self.inputs
        self.weights += delta_weights
        return

    def train_network(self):
        inputs = self.inputs
        self.update_weights()
        weights = self.weights
        print(f"Inputs: {inputs}")
        print(f"Weights: {weights}")
        return


def modify_reinforcement(current_reinforcement):
    modified_rates = current_reinforcement
    modified_rates.reverse()
    return modified_rates


def main():
    perceptrons = []
    for _ in range(10):
        perceptron = Perceptron(4, 0.1, 1)
        perceptrons.append(perceptron)
    for _, perceptron in enumerate(perceptrons):
        perceptron.train_network()
        print(perceptron.compute_output())

    return 0


if __name__ == "__main__":
    main()
