import numpy as np
import random
from libs.input import get_features_and_targets

def normalize_features_minmax(features):
    transposed = list(zip(*features))

    mins = [min(col) for col in transposed]
    maxs = [max(col) for col in transposed]

    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    normalized = [
        tuple(normalize_value(val, mins[i], maxs[i]) for i, val in enumerate(sample))
        for sample in features
    ]
    print(normalized[0])
    return normalized

class Perceptron():
    def __init__(self, activation, inputs_size, bias=0.0):
        self.weights = [random.uniform(-1, 1) for _ in range(inputs_size)]
        self.activation = activation
        self.bias = bias
        self.last_z = 0.0
        self.last_output = 0.0

    def activate(self, x):
        if self.activation == 'sig':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.max(0, x)
        
    def compute_output(self, inputs: tuple) -> float:
        soma = 0
        for x, peso in zip(inputs, self.weights):
            soma += x * peso
        soma += self.bias
        self.last_z = soma
        self.last_output = self.activate(soma)
        return self.last_output
    

class MLP_Network():
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.layers: list[list[Perceptron]] = []
        self.activation = None
        self.results = []

    """
    Cria Perceptrons da rede

    Args:
        inputs_size: número de features
        n_hidden = número de neuronios em cada camada oculta
        n_output = numero de neuronios na camada de saida
        hidden_layers_size = numero de camadas ocultas
        bias = bias inicial dos neuronios
        activation = funcao de ativacao
    """
    def create_network(self, inputs_size, n_hidden, n_output, hidden_layers_size, bias, activation):
        current_inputs_size = inputs_size
        for _ in range(hidden_layers_size):
            layer = [Perceptron(activation, current_inputs_size, bias)
                        for _ in range(n_hidden)]
                
            self.layers.append(layer)
            current_inputs_size = n_hidden

        output_layer = [Perceptron(activation, current_inputs_size, bias)
                        for _ in range(n_output)]
        
        self.layers.append(output_layer)
        self.activation = activation

    def activation_derivative(self, x):
        if self.activation == 'sig':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation == 'relu':
            return 1 if x > 0 else 0
        
    def calc_error(self, expected, predicted):
        return sum((float(e) - float(p)) ** 2 for e, p in  zip(expected, predicted))
        
    def feed_forward(self, inputs: list):
        current_inputs = inputs
        for layer in self.layers:
            next_inputs = []
            for p in layer:
                out = p.compute_output(current_inputs)
                next_inputs.append(out)

            current_inputs = next_inputs

        return current_inputs

    def back_propagation(self, example, predicted_output, expected_output, learning_rate):
        deltas = [None] * len(self.layers)

        output_layer = self.layers[-1]
        output_deltas = []
        for i, perceptron in enumerate(output_layer):
            error = expected_output - predicted_output[i]
            delta = error * self.activation_derivative(perceptron.last_z)
            output_deltas.append(delta)
        deltas[-1] = output_deltas

        for layer_index in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            current_deltas = []
            for j, perceptron in enumerate(current_layer):
                erro = sum(
                    deltas[layer_index + 1][k] * next_layer[k].weights[j]
                    for k in range(len(next_layer))
                )
                delta = erro * self.activation_derivative(perceptron.last_z)
                current_deltas.append(delta)
            deltas[layer_index] = current_deltas

        inputs = example
        for l, layer in enumerate(self.layers):
            for p, perceptron in enumerate(layer):
                if l > 0:
                    inputs = [perceptron_prev.last_output 
                            for perceptron_prev in self.layers[l - 1]]
                for j in range(len(perceptron.weights)):
                    perceptron.weights[j] += learning_rate * deltas[l][p] * inputs[j]
                perceptron.bias += learning_rate * deltas[l][p]


    def fit(self, features: list[tuple], targets: list, learning_rate: float, epochs: int):
        epoch = 0
        features = normalize_features_minmax(features)
        while epoch < epochs:
            total_error = 0
            for i, example in enumerate(features):
                expected_output = targets[i]
                predicted_output = self.feed_forward(list(example))

                fit_error = self.calc_error([expected_output], predicted_output)
                total_error += fit_error
                self.back_propagation(example, predicted_output, expected_output, learning_rate)
                   
            epoch += 1
    
        return total_error
    
    def predict(self, examples: list):
        features, targets = get_features_and_targets(examples, self.task_type)
        predictions = []
        for example, input_vector in zip(examples, features):
            output = self.feed_forward(input_vector)
            if self.task_type == "classifier":
                prediction = output[0]
            else:
                prediction = output[0]
            predictions.append((example, prediction))

        return predictions
