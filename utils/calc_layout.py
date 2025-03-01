import networkx as nx
import numpy as np
from fa2_modified import ForceAtlas2

# List of available layout algorithms
AVAILABLE_LAYOUTS = [
    "hierarchical_betweeness_centrality",
    "connection_centric",
    "weighted_spring",
    "spring",
    "circular",
    "kamada_kawai",
    "spiral",
    "force_atlas2",
    "radial",
    "weighted_spectral",
    "pagerank_centric",
]

AVAILABLE_LAYOUTS_NON_WEIGHTED = [
    "hierarchical_betweeness_centrality",
    "spring",
    "circular",
    "kamada_kawai",
    "spiral",
    "force_atlas2",
    "pagerank_centric",
]


def calc_spring_layout(G):
    """Basic spring layout with weight influence"""
    return nx.spring_layout(G, seed=42, weight="weight", k=0.3)


def calc_circular_layout(G):
    """Circular layout arrangement"""
    return nx.circular_layout(G)


def calc_kamada_kawai_layout(G):
    """Kamada-Kawai force-directed layout"""
    return nx.kamada_kawai_layout(G, weight="weight")


def calc_spectral_layout(G):
    """Spectral layout using graph Laplacian"""
    return nx.spectral_layout(G, weight="weight")


def calc_shell_layout(G):
    """Shell layout arrangement"""
    return nx.shell_layout(G)


def calc_spiral_layout(G):
    """Spiral layout arrangement"""
    return nx.spiral_layout(G)


def calc_force_atlas2_layout(G):
    """ForceAtlas2 layout algorithm"""
    try:
        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=True,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            verbose=False,
        )

        initial_pos = {
            node: np.array([0.5, 0.5]) + np.random.random(2) * 0.1 for node in G.nodes()
        }
        positions = forceatlas2.forceatlas2_networkx_layout(
            G, pos=initial_pos, iterations=100
        )

        pos = {k: np.array([v[0], v[1]]) for k, v in positions.items()}

        # Scale positions to be centered
        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        for k in pos:
            pos[k][0] = (pos[k][0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            pos[k][1] = (pos[k][1] - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            pos[k] = pos[k] * 2 - 1  # Scale to [-1, 1]
        return pos
    except Exception as e:
        print(f"ForceAtlas2 failed: {e}")
        return nx.spring_layout(G, seed=42)


def calc_radial_layout(G):
    """Radial layout based on node degrees"""
    degrees = dict(G.degree(weight="weight"))
    center_node = max(degrees, key=degrees.get)
    pos = nx.kamada_kawai_layout(G, weight="weight")

    center_pos = pos[center_node]
    for node in pos:
        pos[node] = pos[node] - center_pos

    for node in pos:
        if node != center_node:
            dist = np.sqrt(pos[node][0] ** 2 + pos[node][1] ** 2)
            if G.has_edge(node, center_node):
                weight = G[node][center_node]["weight"]
                scale_factor = 1.0 / (weight + 1)
                pos[node] = pos[node] * scale_factor
    return pos


def calc_weighted_spring_layout(G):
    """Spring layout with stronger weight influence"""
    for u, v, d in G.edges(data=True):
        d["weight"] = d["weight"] * 2
    return nx.spring_layout(G, seed=42, weight="weight", k=0.2, iterations=100)


def calc_weighted_spectral_layout(G):
    """Weighted spectral layout using Laplacian eigenvectors"""
    A = nx.to_numpy_array(G, weight="weight")
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return {
        i: (eigenvectors[i, 1], eigenvectors[i, 2]) for i in range(G.number_of_nodes())
    }


def calc_hierarchical_betweeness_centrality_layout(G):
    """Hierarchical layout based on betweenness centrality"""
    centrality = nx.betweenness_centrality(G, weight="weight")
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    pos = {}
    total_nodes = len(sorted_nodes)

    for i, (node, _) in enumerate(sorted_nodes):
        angle = (i / (total_nodes - 1) if total_nodes > 1 else 0) * np.pi
        y = 1 - 2 * i / (total_nodes - 1) if total_nodes > 1 else 0
        x = np.sin(angle)
        pos[node] = np.array([x, y])
    return pos


def calc_connection_centric_layout(G):
    """Connection-centric layout with center node focus"""
    weighted_degrees = {
        node: sum(data["weight"] for _, _, data in G.edges(node, data=True))
        for node in G.nodes()
    }
    center_node = max(weighted_degrees, key=weighted_degrees.get)
    pos = {center_node: np.array([0.0, 0.0])}

    unplaced_nodes = list(G.nodes())
    unplaced_nodes.remove(center_node)
    nodes_above = []
    nodes_below = []

    direct_connections = [
        (node, G[center_node][node]["weight"])
        for node in unplaced_nodes
        if G.has_edge(center_node, node)
    ]
    direct_connections.sort(key=lambda x: x[1], reverse=True)

    for i, (node, _) in enumerate(direct_connections):
        if i % 2 == 0:
            nodes_above.append(node)
        else:
            nodes_below.append(node)
        unplaced_nodes.remove(node)

    while unplaced_nodes:
        best_node = None
        best_score = -1
        best_position = "above"

        for node in unplaced_nodes:
            above_score = sum(
                G[node][n]["weight"] if G.has_edge(node, n) else 0
                for n in nodes_above + [center_node]
            )
            below_score = sum(
                G[node][n]["weight"] if G.has_edge(node, n) else 0
                for n in nodes_below + [center_node]
            )

            if above_score > below_score and above_score > best_score:
                best_node = node
                best_score = above_score
                best_position = "above"
            elif below_score > best_score:
                best_node = node
                best_score = below_score
                best_position = "below"

        if best_node is None and unplaced_nodes:
            best_node = unplaced_nodes[0]
            best_position = "above" if len(nodes_above) <= len(nodes_below) else "below"

        if best_position == "above":
            nodes_above.append(best_node)
        else:
            nodes_below.append(best_node)
        unplaced_nodes.remove(best_node)

    for i, node in enumerate(nodes_above):
        y = (i + 1) * (2.0 / (len(nodes_above) + 1)) if nodes_above else 0
        x_offset = (
            0.3 / (G[node][center_node]["weight"] + 1)
            if G.has_edge(node, center_node)
            else 0.5
        )
        x = x_offset if i % 2 == 0 else -x_offset
        pos[node] = np.array([x, y])

    for i, node in enumerate(nodes_below):
        y = -(i + 1) * (2.0 / (len(nodes_below) + 1)) if nodes_below else 0
        x_offset = (
            0.3 / (G[node][center_node]["weight"] + 1)
            if G.has_edge(node, center_node)
            else 0.5
        )
        x = x_offset if i % 2 == 0 else -x_offset
        pos[node] = np.array([x, y])

    return pos


def calc_pagerank_centric_layout(G):
    """PageRank-based hierarchical layout"""
    pagerank_scores = nx.pagerank(G, weight="weight")
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    pos = {}

    levels = min(20, len(G.nodes()))
    nodes_per_level = max(1, len(G.nodes()) // levels)

    for i, (node, _) in enumerate(sorted_nodes):
        level = min(i // nodes_per_level, levels - 1)
        position_in_level = i % nodes_per_level
        total_in_level = min(nodes_per_level, len(G.nodes()) - level * nodes_per_level)

        x = (
            position_in_level / (total_in_level - 1) * 2 - 1
            if total_in_level > 1
            else 0
        )
        y = 1 - 2 * level / (levels - 1) if levels > 1 else 0

        pos[node] = np.array([x, y])
    return pos


def get_layout_position(G, layout_algorithm="spring"):
    """Get node positions based on specified layout algorithm"""
    layout_functions = {
        "spring": calc_spring_layout,
        "circular": calc_circular_layout,
        "kamada_kawai": calc_kamada_kawai_layout,
        "spectral": calc_spectral_layout,
        "shell": calc_shell_layout,
        "spiral": calc_spiral_layout,
        "force_atlas2": calc_force_atlas2_layout,
        "radial": calc_radial_layout,
        "weighted_spring": calc_weighted_spring_layout,
        "weighted_spectral": calc_weighted_spectral_layout,
        "hierarchical_betweeness_centrality": calc_hierarchical_betweeness_centrality_layout,
        "connection_centric": calc_connection_centric_layout,
        "pagerank_centric": calc_pagerank_centric_layout,
    }

    layout_func = layout_functions.get(layout_algorithm, calc_spring_layout)
    return layout_func(G)
