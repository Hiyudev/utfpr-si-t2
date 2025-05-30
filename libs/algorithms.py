from sklearn import neural_network
from libs.input import Exemplo
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
        tuple(
            normalize_value(val, mins[i], maxs[i]) for i, val in enumerate(sample)
        )
        for sample in features
    ]

    return normalized

def neural_network_classifier(examples: list[Exemplo]):
    network = neural_network.MLPClassifier(
        hidden_layer_sizes=(2,2),
        activation="logistic",
        solver="sgd",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=False,
        early_stopping=False,
        validation_fraction=0,
        n_iter_no_change=10,
    )

    features = [
        (example.q_pa, example.pulso, example.respiracao)
        for example in examples
    ]

    labels = [example.rotulo for example in examples]

    features = normalize(features)

    network.fit(features, labels)
    return network

def neural_network_regressor(examples: list[Exemplo]):
    network = neural_network.MLPRegressor(
        hidden_layer_sizes=(2,2),
        activation="logistic",
        solver="sgd",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=False,
        early_stopping=False,
        validation_fraction=0,
        n_iter_no_change=10,
    )

    features = [
        (example.q_pa, example.pulso, example.respiracao)
        for example in examples
    ]

    features = normalize(features)

    results = [example.gravidade for example in examples]

    network.fit(features, results)
    return network

    

