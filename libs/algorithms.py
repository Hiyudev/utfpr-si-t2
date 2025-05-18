from sklearn import tree, ensemble
from libs.input import Exemplo
from libs.static import (
    ARG_RANDOM_FOREST_TREES,
    ARG_RANDOM_MAX_FEATURES,
    ARG_RANDOM_MAX_SAMPLES,
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

    features = [
        (example.q_pa, example.pulso, example.respiracao, example.gravidade)
        for example in examples
    ]
    labels = [example.rotulo for example in examples]

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

    features = [
        (example.q_pa, example.pulso, example.respiracao)
        for example in examples
    ]
    results = [example.gravidade for example in examples]

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

    features = [
        (example.q_pa, example.pulso, example.respiracao, example.gravidade)
        for example in examples
    ]
    labels = [example.rotulo for example in examples]

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

    features = [
        (example.q_pa, example.pulso, example.respiracao)
        for example in examples
    ]
    results = [example.gravidade for example in examples]

    decision_tree.fit(features, results)
    return decision_tree