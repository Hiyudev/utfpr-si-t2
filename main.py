import sys

from libs.static import (
    ARG_KFOLD_REPETITIONS,
    ARG_VALIDATION_PERCENTAGE,
)
from libs.algorithms import algorithm_id3, algorithm_random_forest
from libs.data import generate_retencao, generate_kfold
from libs.input import Exemplo, read_data
from typing import Callable


def id3(train_set: list[Exemplo], test_set: list[Exemplo]):
    """
    Algoritmo ID3 para classificação.
    """
    decision_tree = algorithm_id3(train_set)

    results: list[tuple[Exemplo, int]] = []
    for example in test_set:
        features = [example.q_pa, example.pulso, example.respiracao, example.gravidade]
        prediction = decision_tree.predict([features])

        results.append((example, prediction[0]))

    return results


def random_forest(train_set: list[Exemplo], test_set: list[Exemplo]):
    """
    Algoritmo Random Forest para classificação.
    """
    # Implementação do algoritmo Random Forest
    random_forest = algorithm_random_forest(train_set)

    results: list[tuple[Exemplo, int]] = []
    for example in test_set:
        features = [example.q_pa, example.pulso, example.respiracao, example.gravidade]
        prediction = random_forest.predict([features])

        results.append((example, prediction[0]))

    return results


def analyse(
    algorithm: Callable[[list[Exemplo], list[Exemplo]], list[tuple[Exemplo, int]]],
    validation: str | None,
) -> None:
    """
    Função para analisar o desempenho do algoritmo.
    """
    examples = read_data("./assets/treino_sinais_vitais_com_label.txt")
    sets: list[tuple[list[Exemplo], list[Exemplo]]] = []

    if validation == "retencao":
        sets = generate_retencao(
            examples, {"ARG_VALIDATION_PERCENTAGE": ARG_VALIDATION_PERCENTAGE}
        )
    elif validation == "kfold":
        sets = generate_kfold(
            examples, {"ARG_KFOLD_REPETITIONS": ARG_KFOLD_REPETITIONS}
        )

    results: list[tuple[Exemplo, int]] = []
    for i, (train_set, test_set) in enumerate(sets):
        local_results: list[tuple[Exemplo, int]] = algorithm(train_set, test_set)

        for example, prediction in local_results:
            results.append((example, prediction))

    # Analisa os resultados
    correct_predictions = 0

    for example, prediction in results:
        if example.rotulo == prediction:
            correct_predictions += 1

    accuracy = correct_predictions / len(results) * 100

    print(f">>> Taxa de erro")
    print(f"Validação: {validation}")

    print("\nResultados:")
    print(f"Total de exemplos: {len(examples)}")
    print(f"Total de validações: {len(results)}")
    print(f"Total de acertos: {correct_predictions}")
    print(f"Total de erros: {len(results) - correct_predictions}")
    print(f"Acurácia: {accuracy:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        algorithm: Callable[
            [list[Exemplo], list[Exemplo]], list[tuple[Exemplo, int]]
        ] = None

        algorithm_arg, validation = sys.argv[1], (
            sys.argv[2] if len(sys.argv) > 2 else None
        )
        if algorithm_arg == "id3":
            algorithm = id3
        elif algorithm_arg == "forest":
            algorithm = id3
        elif algorithm_arg == "redes":
            algorithm = id3
        else:
            print("Argumento inválido. Use 'id3', 'forest' ou 'redes'.")

        if algorithm:
            analyse(algorithm, validation)
    else:
        print("Argumento inválido. Use 'id3', 'forest' ou 'redes'.")
