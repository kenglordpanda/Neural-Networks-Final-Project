import matplotlib.pyplot as plt
import numpy as np

"""
Research paper does not show that bias is implemented so it is not included in this code
"""


class Perceptron:
    def __init__(self, num_inputs, learning_rate):
        self.weights = np.random.uniform(-0.1, 0.1, (num_inputs, 1))
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.prev_responses = []

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def compute_response(self, inputs):
        output_layer_in = self.bias + (self.weights.T @ inputs)
        return self.sigmoid(output_layer_in)

    def update_weights(self, inputs, target):
        delta_weights = np.zeros(self.weights.size)
        network_response = self.compute_response(inputs)
        generated_response = np.random.rand()
        if generated_response <= network_response:
            delta_weights = self.learning_rate * target * inputs
        self.weights += delta_weights


def generate_specific_epochs(
    num_stimuli, epoch_start, epoch_end, reinforcement_probabilities
):
    if num_stimuli != len(reinforcement_probabilities):
        return None
    range = epoch_end - epoch_start
    list_sizes = np.floor(np.array(reinforcement_probabilities) * range).astype(int)
    stimuli_reinforcements = [
        set(
            np.random.choice(
                np.arange(epoch_start, epoch_end + 1), size=size, replace=False
            )
        )
        for size in list_sizes
    ]
    return stimuli_reinforcements


def modify_reinforcement(current_reinforcement):
    modified_rates = current_reinforcement
    modified_rates.reverse()
    return modified_rates


def simulate_bandit(
    num_inputs, num_networks, num_epochs, init_reinforcement, learning_rate
):
    reinforcement_rates = init_reinforcement
    perceptrons = []
    presenations = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    for _ in range(num_networks):
        perceptrons.append(Perceptron(num_inputs, learning_rate))

    for epoch in range(num_epochs):
        if epoch >= num_epochs // 2:
            reinforcement_rates = modify_reinforcement(reinforcement_rates)

    return


def main():
    num_stimuli = 4
    num_perceptrons = 10
    num_epochs = 600
    reinforcement = [0.2, 0.4, 0.6, 0.8]
    learning_rate = 0.1
    simulate_bandit(
        num_stimuli, num_perceptrons, num_epochs, reinforcement, learning_rate
    )
    example_reinforcement = generate_specific_epochs(4, 0, 300, reinforcement)
    for i, s in enumerate(example_reinforcement):
        print(f"Set {i + 1}: {sorted(s)}")
    return 0


if __name__ == "__main__":
    main()
