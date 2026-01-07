from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

from robustness.domain.psf.model import is_terminal
from robustness.domain.tree.model import BinaryTree


def remove_unconnected_nodes(tree: BinaryTree, root: int) -> BinaryTree:
    visited = set()

    for node in nx.dfs_postorder_nodes(tree, root):
        visited.add(node)

    new_tree = tree.copy()
    new_tree.remove_nodes_from(tree.nodes - visited)

    return new_tree


def map_nodes_of(nodes_map: dict[Any, Any], root: Any | None, tree: BinaryTree) -> BinaryTree:
    new_tree = tree.copy()
    new_tree.remove_edges_from(tree.edges)

    frontier = deque([root])
    while frontier:
        current = frontier.popleft()
        attr = tree.nodes[current]

        if is_terminal(attr['kind']):
            continue

        for child in tree.successors(current):
            old_edge_attr = {}
            for edge_p, edge_c, edge_attr in tree.edges(current, data=True):
                if edge_c == child:
                    old_edge_attr = edge_attr.copy()
                    break
            new_child = nodes_map[child]
            frontier.append(new_child)
            new_tree.add_edge(current, new_child, **edge_attr)

    return remove_unconnected_nodes(new_tree, root)
