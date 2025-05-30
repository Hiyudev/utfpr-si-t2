import numpy as np
import random

from libs.input import Exemplo


def normalize(features: list[tuple]):
    """
    Normaliza os valores das features de um conjunto de dados utilizando a técnica Min-Max.

    Cada feature (coluna) é escalada individualmente para o intervalo [0, 1], de acordo com a fórmula:
        normalized = (valor - mínimo) / (máximo - mínimo)

    Parâmetros:
        features (list of tuple/list): Lista de amostras, onde cada amostra é uma tupla ou lista de valores numéricos.

    Retorna:
        list of tuple: Lista de amostras normalizadas, com os valores das features no intervalo [0, 1].

    Exemplo:
        Entrada: [(2, 10), (4, 20), (6, 30)]
        Saída: [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    """
    transposed = list(zip(*features))

    mins = [min(col) for col in transposed]
    maxs = [max(col) for col in transposed]

    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    normalized = [
        tuple(normalize_value(val, mins[i], maxs[i]) for i, val in enumerate(sample))
        for sample in features
    ]

    return normalized


class Perceptron:
    def __init__(self, activation_function: str, inputs_length: int):
        self.weights = [random.uniform(-1, 1) for _ in range(inputs_length)]
        self.activation = activation_function

        self.last_z = 0.0
        self.last_output = 0.0

    def activate(self, x):
        if self.activation == "sig":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.max(0, x)

    def calc_output(self, inputs: list) -> float:
        # z é somatório ponderado de todas as entradas do perceptron com seus respectivos pesos, mais o vies
        z = 0
        for x, w in zip(inputs, self.weights):
            z += x * w

        self.last_z = z
        self.last_output = self.activate(z)

        return self.last_output


class RedesNeurais:
    def __init__(
        self,
        task_type: str,
        inputs_length: int,
        activation_function: str,
        neurons_per_hidden_layer: int,
        num_hidden_layers: int,
        total_output_neurons: int,
    ):
        self.activation = activation_function
        self.task_type = task_type

        self.layers = self._create_network(
            inputs_length,
            activation_function,
            neurons_per_hidden_layer,
            num_hidden_layers,
            total_output_neurons,
        )

    def _create_network(
        self,
        inputs_length: int,
        activation_function: str,
        neurons_per_hidden_layer: int,
        num_hidden_layers: int,
        total_output_neurons: int,
    ):
        """
        Cria a estrutura da rede neural com as camadas especificadas.

        Args:
            inputs_length (int): número de features de entrada.
            activation_function (str): função de ativação ('sig', 'tanh', 'relu').
            neurons_per_hidden_layer (int): número de neurônios em cada camada oculta.
            num_hidden_layers (int): número de camadas ocultas.
            total_output_neurons (int): número de neurônios na camada de saída.
        """

        layers: list[list[Perceptron]] = []
        current_inputs_size = inputs_length

        for _ in range(num_hidden_layers):
            layer = [
                Perceptron(activation_function, current_inputs_size)
                for _ in range(neurons_per_hidden_layer)
            ]

            layers.append(layer)
            current_inputs_size = neurons_per_hidden_layer

        output_layer = [
            Perceptron(activation_function, current_inputs_size)
            for _ in range(total_output_neurons)
        ]

        layers.append(output_layer)
        return layers

    def activation_derivative(self, x) -> float:
        if self.activation == "sig":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "relu":
            return 1 if x > 0 else 0

    def _calc_error(self, expected, predicted):
        return sum((float(e) - float(p)) ** 2 for e, p in zip(expected, predicted))

    def _feed_forward(self, inputs: list[float]):
        current_inputs = inputs
        for layer in self.layers:
            next_inputs = []
            for perceptron in layer:
                output = perceptron.calc_output(current_inputs)
                next_inputs.append(output)

            current_inputs = next_inputs

        return current_inputs

    def _back_propagation(
        self,
        example: tuple[float, float, float],
        predicted_output: list[float],
        expected_output: list[float],
        learning_rate: float,
    ):
        deltas = [None] * len(self.layers)

        # Retropropagação do erro na camada de saída
        output_layer = self.layers[-1]
        output_deltas = []
        for i, perceptron in enumerate(output_layer):
            error: float = expected_output[i] - predicted_output[i]
            delta: float = error * self.activation_derivative(perceptron.last_z)
            output_deltas.append(delta)
        deltas[-1] = output_deltas

        # Retropropagação do erro nas camadas ocultas
        for layer_index in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            current_deltas = []

            # Para cada perceptron na camada atual, calcula o delta
            for i, perceptron in enumerate(current_layer):
                erro = sum(
                    deltas[layer_index + 1][j] * next_layer[j].weights[i]
                    for j in range(len(next_layer))
                )
                delta = erro * self.activation_derivative(perceptron.last_z)
                current_deltas.append(delta)
            deltas[layer_index] = current_deltas

        # Atualiza os pesos e vieses de cada perceptron
        inputs = example

        # Para cada camada
        for layer_index, layer in enumerate(self.layers):
            # Para cada perceptron na camada
            for j, perceptron in enumerate(layer):
                if layer_index > 0:
                    inputs = [
                        perceptron_prev.last_output
                        for perceptron_prev in self.layers[layer_index - 1]
                    ]

                # Para cada peso da entrada do perceptron
                for i in range(len(perceptron.weights)):
                    perceptron.weights[i] += (
                        learning_rate * inputs[i] * deltas[layer_index][j]
                    )

    def fit(self, examples: list[Exemplo], learning_rate: float, epochs: int):
        features: list[tuple[float, float, float]] = [
            (example.q_pa, example.pulso, example.respiracao) for example in examples
        ]
        features: list[tuple[float, float, float]] = normalize(features)

        targets: list[list] = []
        if self.task_type == "classifier":
            # Para classificacao, os alvos sao vetores do tamanho dos valores unicos
            unique_classes = set(example.rotulo for example in examples)

            class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
            for example in examples:
                target_vector = [0] * len(unique_classes)
                target_vector[class_to_index[example.rotulo]] = 1
                targets.append(target_vector)
        else:
            targets = [[example.gravidade] for example in examples]

        # Algoritmo de treinamento da rede neural
        epoch = 0
        while epoch < epochs:
            # Para cada exemplo, calcula a saida e ajusta os pesos
            print(f"Epoch {epoch + 1}/{epochs}")

            for i in range(len(examples)):
                example = features[i]
                expected_output = targets[i]
                predicted_output = self._feed_forward(list(example))

                error = self._calc_error(expected_output, predicted_output)

                print(
                    f"Expected output: {expected_output}, Predicted output: {predicted_output}, Error: {error:.4f}"
                )

                self._back_propagation(
                    example, predicted_output, expected_output, learning_rate
                )

            epoch += 1

    def predict(self, examples: list[Exemplo]):
        features: list[tuple[float, float, float]] = [
            (example.q_pa, example.pulso, example.respiracao) for example in examples
        ]

        predictions = []
        for example, input_vector in zip(examples, features):
            output = self._feed_forward(input_vector)
            prediction = None

            if self.task_type == "classifier":
                # Para classificacao, escolhe a classe com maior probabilidade
                prediction_index = np.argmax(output)

                prediction = prediction_index + 1
            else:
                # Para regressao, simplesmente retorna o valor previsto
                prediction = output[0]
            predictions.append((example, prediction))

        return predictions
