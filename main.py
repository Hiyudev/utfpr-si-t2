import sys

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
    algorithm_id3_classifier,
    algorithm_id3_regressor,
    algorithm_random_forest_classifier,
    algorithm_random_forest_regressor,
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
    redes_regressor = algorithm_neural_network(train_set, 'regressor')
    redes_classifier = algorithm_neural_network(train_set, 'classifier')

    regressor_results: list[tuple[Exemplo, float]] = []
    classifier_results: list[tuple[Exemplo, int]] = []

    # for example in test_set:
    #     classifier_features = [
    #         example.q_pa, example.pulso, example.respiracao, example.gravidade
    #     ]
    #     classifier_prediction = redes_classifier.predict([classifier_features])
    #     classifier_results.append((example, classifier_prediction[0]))
        
    #     regressor_features = [example.q_pa, example.pulso, example.respiracao]
    #     regressor_prediction = redes_regressor.predict([regressor_features])
    #     regressor_results.append((example, regressor_prediction[0]))

    # results: tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]] = [
    #     regressor_results,
    #     classifier_results,
    # ]
    # return results


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

    sets: list[tuple[list[Exemplo], list[Exemplo]]] = []
    if validation == "retencao":
        sets = generate_retencao(
            examples, {"ARG_VALIDATION_PERCENTAGE": ARG_VALIDATION_PERCENTAGE}
        )
    elif validation == "kfold":
        sets = generate_kfold(
            examples, {"ARG_KFOLD_REPETITIONS": ARG_KFOLD_REPETITIONS}
        )

    classifier_results: list[tuple[Exemplo, int]] = []
    regressor_results: list[tuple[Exemplo, float]] = []

    for i, (train_set, test_set) in enumerate(sets):
        local_results: tuple[list[tuple[Exemplo, float]], list[tuple[Exemplo, int]]] = (
            algorithm(train_set, test_set)
        )

        classifier_results.extend(local_results[0])
        regressor_results.extend(local_results[1])

    # Analisa os resultados
    correct_predictions = 0

    for example, prediction in classifier_results:
        if example.rotulo == prediction:
            correct_predictions += 1

    accuracy = correct_predictions / len(classifier_results) * 100

    print(f"Algoritmo: {algorithm.__name__}")
    print(f"Validação: {validation}")
    print(f"Acurácia: {accuracy:.2f}%")

    if regressor_results:
        print(
            f"Diferença quadrática média: {sum((example.gravidade - prediction) ** 2 for example, prediction in regressor_results) / len(regressor_results):.2f}"
        )


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
