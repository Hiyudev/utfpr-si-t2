import numpy as np

from libs.input import Exemplo
from .decision_tree import DecisitionTree


class RandomForest:
    def __init__(
        self,
        estimators: int,
        split_criteria: str,
        examples_bagging: float,
        features_bagging: float,
    ):
        self.estimators = estimators
        self.split_criteria = split_criteria

        self.examples_bagging_perc = examples_bagging
        self.features_bagging_perc = features_bagging

        self.trees: list[DecisitionTree] = []

        for _ in range(self.estimators):
            tree = DecisitionTree(
                split_criteria=self.split_criteria,
            )
            self.trees.append(tree)

    def fit(self, examples: list[Exemplo]):
        features = ["q_pa", "pulso", "respiracao"]
        
        # Realiza o bootstrap para criar subconjuntos de exemplos
        sub_examples = []
        sub_features = []

        examples_bagging_size = int(len(examples) * self.examples_bagging_perc)
        # O numero "3" e o numero de atributos que estamos considerando no treino
        features_bagging_size = int(3 * self.features_bagging_perc)

        for _ in range(self.estimators):
            subset_examples = np.random.choice(
                examples,
                size=examples_bagging_size,
                replace=True,
            )

            subset_features = np.random.choice(
                features,
                size=features_bagging_size,
                replace=False,
            )

            sub_examples.append(subset_examples)
            sub_features.append(subset_features)

        for i in range(self.estimators):
            self.trees[i].fit(sub_examples[i], sub_features[i])

    def predict(self, examples: list[Exemplo]):
        predictions = []

        # Itera sobre cada arvore e agrega as previsoes
        trees_predictions = []
        for tree in self.trees:
            tree_predictions = tree.predict(examples)
            trees_predictions.append(tree_predictions)

        # Realiza a votacao majoritaria
        for i in range(len(examples)):
            votes = {}
            for tree_predictions in trees_predictions:
                label = tree_predictions[i]
                if label not in votes:
                    votes[label] = 0
                votes[label] += 1

            # Seleciona o rótulo com mais votos
            majority_label = max(votes, key=votes.get)
            predictions.append(majority_label)

        return predictions
