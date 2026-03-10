from __future__ import annotations

import networkx as nx

from robustness.domain.tree.model import BinaryTree


def remove_unreachable_nodes_from[T: BinaryTree, R: BinaryTree](graph: T, root: int) -> R:
    """

    Clean a directed graph by removing nodes not reachable from the root.
    This is useful after transformations that may leave behind disconnected nodes.

    Args:
        graph: Binary tree to clean up.
        root: Root node ID of the tree.

    Returns: Cleaned binary tree.

    """
    visited = set()

    for node in nx.dfs_postorder_nodes(graph, root):
        visited.add(node)

    new_tree = graph.copy()
    new_tree.remove_nodes_from(graph.nodes - visited)

    return new_tree
