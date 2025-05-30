import sys
import numpy as np
import concurrent.futures

from libs.static import (
    ARG_DISCRE_PULSO,
    ARG_DISCRE_PULSO_GROUPS,
    ARG_DISCRE_QPA,
    ARG_DISCRE_QPA_GROUPS,
    ARG_DISCRE_RESPIRACAO,
    ARG_DISCRE_RESPIRACAO_GROUPS,
    ARG_KFOLD_REPETITIONS,
    ARG_RANDOM_FOREST_CRITERION,
    ARG_RANDOM_FOREST_ESTIMATORS,
    ARG_RANDOM_MAX_EXAMPLES,
    ARG_RANDOM_MAX_FEATURES,
    ARG_TOTAL_REPETITIONS,
    ARG_VALIDATION_PERCENTAGE,
)
from libs.models.decision_tree import DecisitionTree
from libs.models.random_forest import RandomForest

from libs.data import (
    discretize_examples,
    generate_retencao,
    generate_kfold,
    DiscretizationMethod,
)
from libs.algorithms import (
    algorithm_neural_network,
)
from libs.data import generate_retencao, generate_kfold
from libs.input import Exemplo, read_data
from typing import Callable


def id3(train_set: list[Exemplo], test_set: list[Exemplo]):
    """
    Algoritmo ID3.
    """
    decision_tree_classifier = DecisitionTree(
        split_criteria="entropy",
    )
    decision_tree_classifier.fit(train_set, ["q_pa", "pulso", "respiracao"])
    classifier_results: list[tuple[Exemplo, int]] = []

    for example in test_set:
        classifier_prediction = decision_tree_classifier.predict([example])
        classifier_results.append((example, classifier_prediction[0]))

    results: tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]] = [
        classifier_results,
        [],
    ]
    return results


def random_forest(train_set: list[Exemplo], test_set: list[Exemplo]):
    """
    Algoritmo Random Forest.
    """
    # Implementação do algoritmo Random Forest
    random_forest_classifier = RandomForest(
        estimators=ARG_RANDOM_FOREST_ESTIMATORS,
        split_criteria=ARG_RANDOM_FOREST_CRITERION,
        examples_bagging=ARG_RANDOM_MAX_EXAMPLES,
        features_bagging=ARG_RANDOM_MAX_FEATURES,
    )
    random_forest_classifier.fit(train_set)
    classifier_results: list[tuple[Exemplo, int]] = []

    for example in test_set:
        classifier_prediction = random_forest_classifier.predict([example])
        classifier_results.append((example, classifier_prediction[0]))

    results: tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]] = [
        classifier_results,
        [],
    ]
    return results


def redes(train_set: list[Exemplo], test_set: list[Exemplo]):
    """
    Algoritmo Redes Neurais
    """
    redes_regressor = algorithm_neural_network(train_set, "regressor")
    redes_classifier = algorithm_neural_network(train_set, "classifier")

    regressor_results = redes_regressor.predict(test_set)
    classifier_results = redes_classifier.predict(test_set)

    results: tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]] = [
        regressor_results,
        classifier_results,
    ]
    return results


def analyse(
    algorithm: Callable[
        [list[Exemplo], list[Exemplo]],
        tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]],
    ],
    validation: str | None,
) -> None:
    """
    Função para analisar o desempenho do algoritmo.
    """
    examples = read_data("./assets/treino_sinais_vitais_com_label.txt")

    # Discretização dos atributos caso o problema seja de classificação, especialmente para ID3 e Random Forest
    if algorithm.__name__ == "id3" or algorithm.__name__ == "random_forest":
        discretization_attributes = ["q_pa", "pulso", "respiracao"]
        discretization_methods = {
            "q_pa": DiscretizationMethod[ARG_DISCRE_QPA.upper()],
            "pulso": DiscretizationMethod[ARG_DISCRE_PULSO.upper()],
            "respiracao": DiscretizationMethod[ARG_DISCRE_RESPIRACAO.upper()],
        }
        discretization_parameters = {
            "q_pa": {"groups": ARG_DISCRE_QPA_GROUPS},
            "pulso": {"groups": ARG_DISCRE_PULSO_GROUPS},
            "respiracao": {"groups": ARG_DISCRE_RESPIRACAO_GROUPS},
        }
        for attr in discretization_attributes:
            discretize_examples(
                examples,
                attributes=[attr],
                method=discretization_methods[attr],
                parameters=discretization_parameters[attr],
            )

    total_classifier_results: list[tuple[Exemplo, int]] = []
    total_regressor_results: list[tuple[Exemplo, float]] = []

    def process_set(args):
        train_set, test_set = args
        return algorithm(train_set, test_set)

    for k in range(ARG_TOTAL_REPETITIONS):
        print(f"Repetição {k + 1} de {ARG_TOTAL_REPETITIONS}")

        sets: list[tuple[list[Exemplo], list[Exemplo]]] = []
        if validation == "retencao":
            sets = generate_retencao(
                examples, {"ARG_VALIDATION_PERCENTAGE": ARG_VALIDATION_PERCENTAGE}
            )
        elif validation == "kfold":
            sets = generate_kfold(
                examples, {"ARG_KFOLD_REPETITIONS": ARG_KFOLD_REPETITIONS}
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Processa cada conjunto de treino/teste em paralelo
            futures = [executor.submit(process_set, s) for s in sets]
            for future in concurrent.futures.as_completed(futures):
                local_results = future.result()
                # Filter out None predictions for classifier results
                filtered_classifier_results = [
                    (ex, pred) for ex, pred in local_results[0] if pred is not None
                ]
                total_classifier_results.extend(filtered_classifier_results)
                # Filter out None predictions for regressor results
                filtered_regressor_results = [
                    (ex, pred) for ex, pred in local_results[1] if pred is not None
                ]
                total_regressor_results.extend(filtered_regressor_results)

    # Analisa os resultados
    true_positives = [0, 0, 0, 0]
    true_negatives = [0, 0, 0, 0]
    false_positives = [0, 0, 0, 0]
    false_negatives = [0, 0, 0, 0]

    for example, prediction in total_classifier_results:
        # Para cada resultado possivel (0, 1, 2, 3), contabiliza os acertos e erros
        real_label = example.rotulo
        predicted_label = prediction

        if real_label == predicted_label:
            true_positives[real_label - 1] += 1
        elif real_label != predicted_label:
            false_positives[predicted_label - 1] += 1
            false_positives[real_label - 1] += 1
            false_negatives[real_label - 1] += 1
            false_negatives[predicted_label - 1] += 1
        else:
            true_negatives[real_label - 1] += 1

    accuracy = sum(true_positives) / len(total_classifier_results) * 100
    # Precision: average over classes, avoid division by zero
    precisions = []
    for i in range(len(true_positives)):
        tp = true_positives[i]
        fp = false_positives[i]
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
    precision = np.mean(precisions) * 100
    loss = np.mean(
        [
            (example.rotulo - prediction) ** 2
            for example, prediction in total_classifier_results
        ]
    )

    print(f"Algoritmo: {algorithm.__name__}")
    print(f"Validação: {validation}")
    print(f"Acurácia: {accuracy:.2f}%")
    print(f"Precisão: {precision:.2f}%")
    print(f"Perda: {loss:.2f}")

    if len(total_regressor_results) > 0:
        rmse = np.sqrt(
            np.mean(
                [
                    (example.rotulo - prediction) ** 2
                    for example, prediction in total_regressor_results
                ]
            )
        )

        print(f"RMSE: {rmse:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        algorithm: Callable[
            [list[Exemplo], list[Exemplo]],
            tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]],
        ] = None

        algorithm_arg, validation = sys.argv[1], (
            sys.argv[2] if len(sys.argv) > 2 else None
        )
        if algorithm_arg == "id3":
            algorithm = id3
        elif algorithm_arg == "forest":
            algorithm = random_forest
        elif algorithm_arg == "redes":
            algorithm = redes
        else:
            print("Argumento inválido. Use 'id3', 'forest' ou 'redes'.")

        if algorithm:
            analyse(algorithm, validation)
    else:
        print("Argumento inválido. Use 'id3', 'forest' ou 'redes'.")
