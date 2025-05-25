from .decision_tree import DecisionTreeNode
from treelib import Node, Tree


def plot_decision_tree(node: DecisionTreeNode, filename="./output/decision_tree.txt"):
    def build_tree(tree, node, parent=None, edge_label=None):
        label = node.feature if node.feature else str(node.value)
        node_id = id(node)
        node_tag = f"{label}"
        if edge_label is not None:
            node_tag += f" [{edge_label}]"
        tree.create_node(tag=node_tag, identifier=node_id, parent=parent)
        if hasattr(node, "children") and node.children:
            for connector in node.children:
                child = connector.node
                threshold = getattr(connector, "threshold", None)
                if threshold is None and hasattr(connector, "label"):
                    threshold = connector.label
                build_tree(tree, child, parent=node_id, edge_label=threshold)

    tree = Tree()
    build_tree(tree, node)
    tree.save2file(filename)
