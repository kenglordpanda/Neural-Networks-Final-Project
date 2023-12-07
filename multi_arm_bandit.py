import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate):
        # Initialize weights and bias
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def choose_arm(self, inputs):
        # Calculate weighted sum and apply sigmoid activation
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        probability = self.sigmoid(weighted_sum)
        
        # Choose an arm based on the probability
        chosen_arm = np.random.choice([0, 1], p=[1 - probability, probability])
        return chosen_arm

    def update_weights(self, inputs, reward):
        # Update weights and bias based on the obtained reward using gradient descent
        prediction = self.sigmoid(np.dot(self.weights, inputs) + self.bias)
        error = reward - prediction

        # Update weights and bias using the gradient descent rule
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error

def simulate_bandit(num_epochs, num_perceptrons, learning_rate):
    num_inputs = 10
    perceptrons = [Perceptron(num_inputs, learning_rate) for _ in range(num_perceptrons)]
    weights_lists = [[] for _ in range(num_perceptrons)]  # Store weights for each perceptron

    for epoch in range(num_epochs):
        # Determine reinforcement frequencies based on the epoch
        if epoch < num_epochs // 2:
            reinforcement_probabilities = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3, 0.7, 0.9, 0.1, 0.4]
        else:
            reinforcement_probabilities = [0.8, 0.6, 0.4, 0.2, 0.7, 0.9, 0.5, 0.3, 0.9, 0.6]

        for _ in range(10):  # Repeat each pattern ten times
            for i, perceptron in enumerate(perceptrons):
                # Simulate input pattern for the presence of one of the DSs
                inputs = np.array([1 if j == i else 0 for j in range(num_inputs)])

                # Determine the desired response and reinforcement probability
                desired_response = 1 if np.random.rand() < reinforcement_probabilities[i] else 0

                # Choose an arm based on the perceptron's output
                chosen_arm = perceptron.choose_arm(inputs)

                # Update weights based on the obtained reward
                perceptron.update_weights(inputs, desired_response)

        # Store weights for each perception
        for i, perceptron in enumerate(perceptrons):
            weights_lists[i].append(perceptron.weights.copy())

    # Plot weights for each perceptron on separate graphs
    plt.figure(figsize=(15, 10))
    for i in range(num_perceptrons):
        plt.subplot(2, 5, i + 1)
        plt.plot(range(1, num_epochs + 1), weights_lists[i])
        plt.title(f'Perceptron {i + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Value')

    plt.tight_layout()
    plt.show()

# Parameters
num_epochs = 600
num_perceptrons = 10
learning_rate = 0.1

# Simulate the bandit problem
simulate_bandit(num_epochs, num_perceptrons, learning_rate)
