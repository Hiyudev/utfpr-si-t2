from sklearn import tree, ensemble
from libs.input import Exemplo, get_features_and_targets
from libs.static import (
    ARG_RANDOM_FOREST_TREES,
    ARG_RANDOM_MAX_FEATURES,
    ARG_RANDOM_MAX_SAMPLES,
)
from libs.redes import MLP_Network
from libs.static import (
    ACTIVATION_FUNCTION,
    MAX_EPOCHS, 
    BIAS,
    LEARNING_RATE,
    HIDDEN_LAYERS_SIZE,
    NEURONS_OUTPUT_LAYER,
    NEURONS_PER_LAYER
)

def algorithm_id3_classifier(examples: list[Exemplo]) -> tree.DecisionTreeClassifier:
    decision_tree = tree.DecisionTreeClassifier(
        criterion="entropy",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
    )

    features, labels = get_features_and_targets(examples, 'classifier')

    decision_tree.fit(features, labels)
    return decision_tree

def algorithm_id3_regressor(examples: list[Exemplo]) -> tree.DecisionTreeRegressor:
    decision_tree = tree.DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
    )

    features, results = get_features_and_targets(examples, 'regressor')

    decision_tree.fit(features, results)
    return decision_tree


def algorithm_random_forest_classifier(examples: list[Exemplo]) -> tree.DecisionTreeClassifier:
    decision_tree = ensemble.RandomForestClassifier(
        n_estimators=ARG_RANDOM_FOREST_TREES,
        criterion="entropy",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=ARG_RANDOM_MAX_FEATURES,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        max_samples=ARG_RANDOM_MAX_SAMPLES,
    )

    features, labels = get_features_and_targets(examples, 'classifier')

    decision_tree.fit(features, labels)
    return decision_tree

def algorithm_random_forest_regressor(examples: list[Exemplo]) -> tree.DecisionTreeRegressor:
    decision_tree = ensemble.RandomForestRegressor(
        n_estimators=ARG_RANDOM_FOREST_TREES,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=ARG_RANDOM_MAX_FEATURES,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        max_samples=ARG_RANDOM_MAX_SAMPLES,
    )

    features, results = get_features_and_targets(examples, 'regressor')

    decision_tree.fit(features, results)
    return decision_tree

def algorithm_neural_network(examples: list[Exemplo], task_type: str):
    network = MLP_Network(task_type)

    features, targets = get_features_and_targets(examples, task_type)

    if task_type == 'regressor':
        n_output = 1
    else:
        n_output = NEURONS_OUTPUT_LAYER

    network.create_network(inputs_size=len(features[0]),
                           n_hidden=NEURONS_PER_LAYER,
                           n_output=n_output,
                           hidden_layers_size=HIDDEN_LAYERS_SIZE,
                           bias=BIAS,
                           activation=ACTIVATION_FUNCTION)

    network.fit(features, targets, LEARNING_RATE, MAX_EPOCHS)

    return network

    

