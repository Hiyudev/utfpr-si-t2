from typing import Literal, TypeAlias

import numpy as np
import random

from libs.input import Exemplo

mat: TypeAlias = list[list[float]]


class RedesNeurais:
    def __init__(
        self,
        task_type: Literal["classifier", "regressor"],
        arquitecture: list[int],
        activations: list[Literal["sigmoid", "tanh", "relu"]],
    ):
        self.task_type: Literal["classifier", "regressor"] = task_type
        self.layers = len(arquitecture)
        self.architecture = arquitecture

        # Por exemplo, activations[0] é a função de ativação da camada 1
        self.activations = activations

        # Representação dos pesos como uma lista de matrizes. Cada matriz representa todos os pesos de uma camada para a próxima.
        # Por exemplo, weights[0] contém os pesos da camada 0 para a camada 1, weights[1] contém os pesos da camada 1 para a camada 2, etc.
        self.weights = [
            np.random.randn(y, x) for x, y in zip(arquitecture[:-1], arquitecture[1:])
        ]

    def fit(
        self,
        examples: list[Exemplo],
        learning_rate: float,
        subset_size: int,
        epochs: int,
    ):
        # Algoritmo de treinamento da rede neural
        for epoch in range(epochs):
            random.shuffle(
                examples
            )  # Embaralha os exemplos a cada época de treinamento

            subsets = [
                examples[i : i + subset_size]
                for i in range(0, len(examples), subset_size)
            ]

            for subset in subsets:
                self._fit(subset, learning_rate)

    def _fit(self, examples: list[Exemplo], learning_rate: float):
        features: list[tuple[float, float, float]] = [
            (example.q_pa, example.pulso, example.respiracao) for example in examples
        ]
        features = self._normalize(features)

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

        for example, expected_output in zip(features, targets):
            predicted_output: list[float] = self._feed_forward(example)

            print(f"Erro: {self._calc_error(expected_output, predicted_output):.4f} ")

            self._back_propagate(
                example, predicted_output, expected_output, learning_rate
            )

    def _back_propagate(
        self,
        example: tuple[float, float, float],
        predicted_output: list[float],
        expected_output: list[float],
        learning_rate: float,
    ):
        input = list(example)
        # Inicializa uma matriz de deltas para armazenar os erros de cada perceptron de cada camada
        deltas: mat = [
            [0.0] * self.architecture[layer_index]
            for layer_index in range(self.layers - 1)
        ]

        # Calcula todas as entradas/saídas da rede
        activation = input
        activations: list[list[float]] = [activation]

        last_layer_index = self.layers - 1
        for i in range(last_layer_index):
            activation = self._get_output(activation, i)
            activations.append(activation)

        # Propaga os deltas para trás
        # Para cada neurônio da última camada, calcula o delta
        for i in range(self.architecture[last_layer_index]):
            local_err = expected_output[i] - predicted_output[i]
            local_input = activations[-1][i]

            deltas[-1][i] = (
                self._get_activation_derivative(local_input, self.activations[-1])
                * local_err
            )

        # Para cada camada, exceto a última, calcula o delta
        for layer_index in range(self.layers - 2, -1, -1):
            current_layer = self.weights[layer_index]
            next_layer = self.weights[layer_index + 1]

            # Para cada perceptron na camada atual, calcula o delta
            for i in range(self.architecture[layer_index]):
                # Calcula o erro para o perceptron atual
                erro = sum(
                    deltas[layer_index + 1][j] * next_layer[j][i]
                    for j in range(self.architecture[layer_index + 1])
                )
                local_input = activations[layer_index][i]

                # Calcula o delta
                deltas[layer_index][i] = (
                    self._get_activation_derivative(
                        local_input, self.activations[layer_index]
                    )
                    * erro
                )

        # Atualiza os pesos
        for layer_index in range(self.layers - 1):
            current_layer = self.weights[layer_index]
            next_layer = self.weights[layer_index + 1]

            for i in range(self.architecture[layer_index]):
                for j in range(self.architecture[layer_index + 1]):
                    # Atualiza o peso com base no delta e na taxa de aprendizado
                    current_layer[i][j] += (
                        learning_rate
                        * deltas[layer_index + 1][j]
                        * activations[layer_index][i]
                    )

            # Atualiza os pesos da camada atual
            self.weights[layer_index] = current_layer

    def predict(self, examples: list[Exemplo]) -> list[float] | list[int]:
        features: list[tuple[float, float, float]] = [
            (example.q_pa, example.pulso, example.respiracao) for example in examples
        ]
        predictions: list[float] | list[int] = []

        for example in features:
            output: list[float] = self._feed_forward(example)

            if self.task_type == "classifier":
                # Para classificação, retorna o índice do valor máximo
                predictions.append(output.index(max(output)))
            else:
                # Para regressão, retorna o valor diretamente
                predictions.append(output)

        return predictions

    def _get_activation(self, value: float, activation: str) -> float:
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-value))
        elif activation == "tanh":
            return np.tanh(value)
        elif activation == "relu":
            return max(0, value)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def _get_output(self, inputs: list[float], current_layer_index: int) -> list[float]:
        """
        A partir dos inputs e do índice da camada atual, calcula a saída da camada. A saída intuitivamente é a entrada da próxima camada.
        """
        output = []

        weights: mat = self.weights[current_layer_index]
        for weight in weights:
            escalar: float = np.dot(weight, inputs)
            activation_function: str = self.activations[current_layer_index]
            activation_output: float = self._get_activation(
                escalar, activation_function
            )

            output.append(activation_output)

        return output

    def _feed_forward(self, inputs: list[float]) -> list[float]:
        current_inputs = inputs

        for i in range(self.layers - 1):
            current_inputs = self._get_output(current_inputs, i)

        return current_inputs

    def _normalize(self, features: list[tuple]):
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
            tuple(
                normalize_value(val, mins[i], maxs[i]) for i, val in enumerate(sample)
            )
            for sample in features
        ]

        return normalized

    def _calc_error(
        self,
        expected_output: list[float],
        predicted_output: list[float],
    ) -> float:
        """
        Calcula o erro quadrático médio entre a saída esperada e a saída prevista.
        """
        return sum(
            (float(e) - float(p)) ** 2
            for e, p in zip(expected_output, predicted_output)
        )

    def _get_activation_derivative(self, value: float, activation: str) -> float:
        if activation == "sigmoid":
            sig = 1 / (1 + np.exp(-value))
            return sig * (1 - sig)
        elif activation == "tanh":
            return 1 - np.tanh(value) ** 2
        elif activation == "relu":
            return 1 if value > 0 else 0
        else:
            raise ValueError(f"Unknown activation function: {activation}")
