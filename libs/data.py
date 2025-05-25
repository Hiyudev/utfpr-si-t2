from libs.input import Exemplo

from random import choice
from typing import Literal, TypeVar, List
from sklearn.model_selection import RepeatedKFold
from skimage.filters import threshold_multiotsu
import numpy as np
from enum import Enum


def generate_retencao(
    examples: list[Exemplo], parameters: dict
) -> list[tuple[list[Exemplo], list[Exemplo]]]:
    copy_examples = examples.copy()
    examples_length = len(copy_examples)

    # Porcentagem de dados utilizados para o conjunto de validação, o restante será utilizado para o treinamento
    ARG_VALIDATION_PERCENTAGE = parameters.get("ARG_VALIDATION_PERCENTAGE", 0.2)
    validation_length = int(examples_length * ARG_VALIDATION_PERCENTAGE)

    training_set: list[Exemplo] = []
    validation_set: list[Exemplo] = []

    for i in range(examples_length):
        # Escolhe um exemplo aleatório para o conjunto de validação
        random_example = choice(copy_examples)
        copy_examples.remove(random_example)

        # Adiciona ao conjunto de validação se a porcentagem de validação não for atingida
        if len(validation_set) < validation_length:
            validation_set.append(random_example)
        else:
            training_set.append(random_example)

    return [[training_set, validation_set]]


def generate_kfold(
    examples: list[Exemplo], parameters: dict
) -> list[tuple[list[Exemplo], list[Exemplo]]]:
    """
    Gera o conjunto de treinamento e validação para o algoritmo de K-Fold.
    """
    folds: list[tuple[list[Exemplo], list[Exemplo]]] = []

    # O numero de folds
    k = parameters.get("ARG_KFOLD_REPETITIONS", 5)

    rkf = RepeatedKFold(n_splits=k, n_repeats=k, random_state=None)
    for train_index, val_index in rkf.split(examples):
        train_set = [examples[i] for i in train_index]
        val_set = [examples[i] for i in val_index]
        folds.append((train_set, val_set))

    return folds


T = TypeVar("T")


def generate_discretized_data_otsu(data: List[T], parameters: dict):
    reshaped_data = np.array(data).reshape(-1, 1)
    classes = parameters.get("groups", 3)

    thresholds = threshold_multiotsu(reshaped_data, classes=classes)

    # Cria listas vazias para cada grupo baseado no número de thresholds + 1
    groups = [[] for _ in range(len(thresholds) + 1)]
    discretized_data = []

    for el in data:
        # Encontra o índice do grupo correto para o elemento
        idx = 0
        while idx < len(thresholds) and el >= thresholds[idx]:
            idx += 1
        groups[idx].append(el)
        discretized_data.append(idx)

    return groups, thresholds, discretized_data


def generate_discretized_data_percentil(data: List[T], parameters: dict):
    classes = parameters.get("groups", 3)
    percentiles = np.percentile(data, np.linspace(0, 100, classes + 1)[1:-1])

    groups = [[] for _ in range(classes)]
    discretized_data = []

    for el in data:
        idx = 0
        while idx < len(percentiles) and el >= percentiles[idx]:
            idx += 1
        groups[idx].append(el)
        discretized_data.append(idx)

    return groups, percentiles, discretized_data


def generate_discretized_data_normal(data: List[T], parameters: dict):
    classes = parameters.get("groups", 3)

    mean = np.mean(data)
    std = np.std(data)

    # Calcula os limites para dividir os dados em 'classes' grupos igualmente espaçados em torno da média
    # Exemplo para 3 grupos: limites = [mean - std, mean + std]
    half = classes // 2
    if classes % 2 == 1:
        # Para número ímpar de classes, centraliza na média
        limits = [mean + std * (i - half) for i in range(1, classes)]
    else:
        # Para número par de classes, centraliza entre valores
        limits = [mean + std * (i - half + 0.5) for i in range(1, classes)]

    groups = [[] for _ in range(classes)]
    discretized_data = []

    for el in data:
        idx = 0
        while idx < len(limits) and el >= limits[idx]:
            idx += 1
        groups[idx].append(el)
        discretized_data.append(idx)

    return groups, limits, discretized_data


class DiscretizationMethod(Enum):
    OTSU = "otsu"
    PERCENTIL = "percentil"
    NORMAL = "normal"


def generate_discretized_data(
    data: List[T],
    method: DiscretizationMethod,
    parameters: dict,
):
    groups = None
    limits = None
    discretized_data = None

    if method == DiscretizationMethod.OTSU:
        local_groups, local_limits, local_discretized = generate_discretized_data_otsu(
            data, parameters
        )
        groups = local_groups
        limits = local_limits
        discretized_data = local_discretized
    elif method == DiscretizationMethod.PERCENTIL:
        local_groups, local_limits, local_discretized = (
            generate_discretized_data_percentil(data, parameters)
        )
        groups = local_groups
        limits = local_limits
        discretized_data = local_discretized
    elif method == DiscretizationMethod.NORMAL:
        local_groups, local_limits, local_discretized = (
            generate_discretized_data_normal(data, parameters)
        )
        groups = local_groups
        limits = local_limits
        discretized_data = local_discretized
    else:
        raise ValueError(
            "Método inválido. Escolha entre 'otsu', 'percentil' ou 'normal'."
        )

    return groups, limits, discretized_data


def discretize_examples(
    examples: List[Exemplo],
    attributes: List[str],
    method: DiscretizationMethod = DiscretizationMethod.OTSU,
    parameters: dict = {},
):
    """
    Discretiza os atributos dos exemplos utilizando o método especificado.
    """

    for attribute in attributes:
        if not all(hasattr(example, attribute) for example in examples):
            raise ValueError(
                f"Atributo '{attribute}' não encontrado em todos os exemplos da classe Exemplo."
            )

        discretized_attribute = [getattr(example, attribute) for example in examples]
        groups, thresholds, discretized = generate_discretized_data(
            discretized_attribute, method=method, parameters=parameters
        )

        for i, example in enumerate(examples):
            setattr(example, attribute, discretized[i])
