from random import choice
from sklearn.model_selection import RepeatedKFold
from libs.input import Exemplo

def generate_retencao(examples: list[Exemplo], parameters: dict) -> list[tuple[list[Exemplo], list[Exemplo]]]:
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

def generate_kfold(examples: list[Exemplo], parameters: dict) -> list[tuple[list[Exemplo], list[Exemplo]]]:
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