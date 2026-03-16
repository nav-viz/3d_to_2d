"""Graph utility functions for the CAD Drawing Generator."""
from typing import Any, Dict, List, Optional, Set, Tuple


class GraphUtils:
    """Helper methods for graph operations on BRep topology."""

    @staticmethod
    def build_adjacency_list(
        edges: List[Tuple[int, int]],
        num_nodes: int,
    ) -> Dict[int, List[int]]:
        """
        Build an adjacency list from an edge list.

        Args:
            edges: List of (source, target) integer index pairs.
            num_nodes: Total number of nodes.

        Returns:
            Dictionary mapping each node index to its neighbour indices.
        """
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        for src, tgt in edges:
            if 0 <= src < num_nodes and 0 <= tgt < num_nodes:
                if tgt not in adj[src]:
                    adj[src].append(tgt)
                if src not in adj[tgt]:
                    adj[tgt].append(src)
        return adj

    @staticmethod
    def compute_degree_sequence(adj: Dict[int, List[int]]) -> List[int]:
        """Return the sorted degree sequence for the graph."""
        return sorted([len(neighbours) for neighbours in adj.values()], reverse=True)

    @staticmethod
    def bfs(
        start: int, adj: Dict[int, List[int]]
    ) -> List[int]:
        """
        Breadth-first traversal from *start*.

        Returns:
            List of visited node indices in BFS order.
        """
        visited: Set[int] = set()
        queue = [start]
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    queue.append(nb)
        return order

    @staticmethod
    def connected_components(adj: Dict[int, List[int]]) -> List[List[int]]:
        """Return all connected components as lists of node indices."""
        visited: Set[int] = set()
        components: List[List[int]] = []
        for node in adj:
            if node not in visited:
                component = GraphUtils.bfs(node, adj)
                visited.update(component)
                components.append(component)
        return components

    @staticmethod
    def compute_graph_diameter(adj: Dict[int, List[int]]) -> int:
        """
        Estimate the graph diameter using BFS from each node.

        For large graphs this may be slow; used only on small BRep graphs.
        """
        if not adj:
            return 0
        diameter = 0
        nodes = list(adj.keys())
        sample = nodes[:min(len(nodes), 20)]  # cap for large graphs
        for start in sample:
            depths = GraphUtils._bfs_depths(start, adj)
            if depths:
                diameter = max(diameter, max(depths.values()))
        return diameter

    @staticmethod
    def _bfs_depths(start: int, adj: Dict[int, List[int]]) -> Dict[int, int]:
        """Return BFS depths from *start* to all reachable nodes."""
        depths: Dict[int, int] = {start: 0}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in adj.get(node, []):
                if nb not in depths:
                    depths[nb] = depths[node] + 1
                    queue.append(nb)
        return depths

    @staticmethod
    def node_centrality(adj: Dict[int, List[int]]) -> Dict[int, float]:
        """
        Compute simple degree centrality for each node.

        centrality(v) = degree(v) / (|V| - 1)
        """
        n = len(adj)
        if n <= 1:
            return {k: 0.0 for k in adj}
        return {node: len(neighbours) / (n - 1) for node, neighbours in adj.items()}

    @staticmethod
    def find_cycles(adj: Dict[int, List[int]]) -> bool:
        """Return True if the undirected graph contains at least one cycle."""
        visited: Set[int] = set()

        def dfs(node: int, parent: Optional[int]) -> bool:
            visited.add(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    if dfs(nb, node):
                        return True
                elif nb != parent:
                    return True
            return False

        for start in adj:
            if start not in visited:
                if dfs(start, None):
                    return True
        return False

    @staticmethod
    def subgraph(
        adj: Dict[int, List[int]],
        node_subset: List[int],
    ) -> Dict[int, List[int]]:
        """Return the induced subgraph over *node_subset*."""
        subset_set = set(node_subset)
        return {
            node: [nb for nb in neighbours if nb in subset_set]
            for node, neighbours in adj.items()
            if node in subset_set
        }

    @staticmethod
    def to_edge_list(adj: Dict[int, List[Any]]) -> List[Tuple[int, int]]:
        """Convert adjacency list to a deduplicated edge list."""
        seen: Set[Tuple[int, int]] = set()
        edges: List[Tuple[int, int]] = []
        for src, neighbours in adj.items():
            for tgt in neighbours:
                key = (min(src, tgt), max(src, tgt))
                if key not in seen:
                    seen.add(key)
                    edges.append((src, tgt))
        return edges
