from numpy import log2
from libs.data import Exemplo
from libs.static import ARG_DECISION_TREE_CRITERION


class DecisionTreeNodeConnector:
    def __init__(self, node=None, threshold=None, feature=None):
        self.node = node
        self.threshold = threshold  # Value at which the split occurs
        self.feature = feature  # Feature index used for the split


class DecisionTreeNode:
    def __init__(self, feature=None, value=None):
        self.feature = feature  # Feature index to split on
        self.value = value  # Value for leaf nodes (if any)

        self.children: list[DecisionTreeNodeConnector] = []  # List of child nodes

    def add_child(self, child: "DecisionTreeNode", threshold, feature):
        connector = DecisionTreeNodeConnector(
            node=child, threshold=threshold, feature=feature
        )
        self.children.append(connector)


class DecisitionTree:
    def __init__(self):
        self.root = None
        self.split_criteria = ARG_DECISION_TREE_CRITERION

    def fit(self, examples: list[Exemplo], attributes: list[str]):
        root = self._build_tree(examples, attributes)
        self.root = root

    def predict(self, examples: list[Exemplo]) -> list[any]:
        predictions = []
        for example in examples:
            predictions.append(self._predict(example, self.root))
        return predictions

    def _build_tree(
        self,
        examples: list[Exemplo],
        attributes: list[str],
        parent_examples: list[Exemplo] = None,
    ):
        # Se nao houver exemplos, retorna o rotulo da maioria
        if self._is_empty(examples):
            return self._get_majority_class(parent_examples)

        # Se todos tem o mesmo rotulo
        if self._is_homogeneous(examples):
            return examples[0].rotulo

        # Se atributos for vazio, retorna o rotulo da maioria
        if self._is_empty(attributes):
            return self._get_majority_class(examples)

        best_attribute = self._best_attribute_to_split(examples, attributes)
        root = DecisionTreeNode(feature=best_attribute)
        unique_values = set(example.features[best_attribute] for example in examples)
        subset_attributes = [attr for attr in attributes if attr != best_attribute]

        for value in unique_values:
            subset_examples = [
                example
                for example in examples
                if example.features[best_attribute] == value
            ]

            subtree_or_label = self._build_tree(
                subset_examples,
                subset_attributes,
                examples,
            )

            if isinstance(subtree_or_label, DecisionTreeNode):
                subtree = subtree_or_label
                root.add_child(subtree, threshold=value, feature=best_attribute)
            else:
                label = subtree_or_label
                leaf_node = DecisionTreeNode(feature=None, value=label)

                root.add_child(leaf_node, threshold=value, feature=best_attribute)

        return root

    def _best_attribute_to_split(
        self,
        examples: list[Exemplo],
        attributes: list[str],
    ):
        selected_attribute = None

        if self.split_criteria == "gini":
            max_importance = -1

            for attribute in attributes:
                importance = self._get_importance(examples, attribute)
                if importance > max_importance:
                    max_importance = importance
                    selected_attribute = attribute
        elif self.split_criteria == "entropy":
            min_importance = float("inf")

            for attribute in attributes:
                importance = self._get_importance(examples, attribute)
                if importance < min_importance:
                    min_importance = importance
                    selected_attribute = attribute

        return selected_attribute

    def _get_importance(
        self,
        examples: list[Exemplo],
        attribute: str,
    ):
        if self.split_criteria == "gini":
            return self._gini_index(examples, attribute)
        elif self.split_criteria == "entropy":
            return self._entropy(examples, attribute)

    def _get_majority_class(self, examples: list[Exemplo]):
        class_counts = {}
        for example in examples:
            label = example.rotulo

            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        return max(class_counts, key=class_counts.get)

    def _is_empty(self, examples: list[Exemplo]):
        return len(examples) == 0

    def _is_homogeneous(self, examples: list[Exemplo]):
        examples_set = set(example.rotulo for example in examples)

        if len(examples_set) != 1:
            return False

        return True

    def _predict(self, example, node):
        if node.value is not None:
            return node.value

        for connector in node.children:
            if example.features[node.feature] == connector.threshold:
                return self._predict(example, connector.node)

        return None

    def _gini_index(self, examples: list[Exemplo], attribute: str):
        class_counts = {}
        for example in examples:
            label = example.rotulo
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        total_count = len(examples)
        gini = 1.0

        for label, count in class_counts.items():
            prob = count / total_count
            gini -= prob**2

        return gini

    def _entropy(self, examples: list[Exemplo], attribute: str):
        class_counts = {}
        for example in examples:
            label = example.rotulo
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        total_count = len(examples)
        entropy = 0.0

        for label, count in class_counts.items():
            prob = count / total_count
            entropy -= prob * log2(prob)

        return entropy
