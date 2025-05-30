import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import SpectralClustering
import scipy.sparse as sparse

# Try to import community module, but provide fallback if not available
try:
    from community import best_partition  # python-louvain package for Louvain algorithm

    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False
    print(
        "Warning: python-louvain package not installed. Using networkx's community detection as fallback."
    )

# Try to import leidenalg module, but provide fallback if not available
try:
    import leidenalg as la
    import igraph as ig

    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    print(
        "Warning: leidenalg package not installed. Using networkx's community detection as fallback."
    )

# Try to import infomap module, but provide fallback if not available
try:
    import infomap

    HAS_INFOMAP = True
except ImportError:
    HAS_INFOMAP = False
    print(
        "Warning: infomap package not installed. Using networkx's community detection as fallback."
    )


def detect_communities_spectral(G, num_clusters=None):
    """
    Detect communities using spectral clustering.
    """
    if num_clusters is None:
        # Estimate number of clusters using eigengap heuristic
        laplacian = nx.normalized_laplacian_matrix(G)
        try:
            # Try newer scipy API first
            eigenvalues = sorted(abs(sparse.linalg.eigvalsh(laplacian.todense())))
        except AttributeError:
            try:
                # Fallback to older API
                eigenvalues = sorted(abs(sparse.linalg.eigvals(laplacian.todense())))
            except:
                # If both fail, use a default number of clusters
                print("Warning: Could not compute eigenvalues, using default number of clusters")
                eigenvalues = []
        
        if len(eigenvalues) > 0:
            gaps = np.diff(eigenvalues)
            num_clusters = np.argmax(gaps) + 2  # Add 2 because we diff'd and want at least 2 communities
        else:
            # Default to square root of number of nodes if eigenvalue computation fails
            num_clusters = max(2, min(int(np.sqrt(len(G.nodes()))), len(G.nodes()) - 1))
            
        num_clusters = max(2, min(num_clusters, len(G.nodes()) - 1))  # Ensure reasonable bounds

    # Create adjacency matrix
    adj_matrix = nx.to_numpy_array(G)

    # Apply spectral clustering
    sc = SpectralClustering(n_clusters=num_clusters, affinity="precomputed", random_state=42)
    labels = sc.fit_predict(adj_matrix)

    # Convert to dictionary format
    return {node: label for node, label in enumerate(labels)}


def detect_communities_infomap(G):
    """
    Detect communities using Infomap algorithm.
    """
    if not HAS_INFOMAP:
        return detect_communities_networkx(G)

    try:
        # Initialize Infomap
        im = infomap.Infomap("--two-level")

        # Add nodes first to ensure all nodes are included
        for node in G.nodes():
            im.add_node(node)

        # Add edges
        for e in G.edges(data=True):
            weight = e[2].get("weight", 1.0)
            im.add_link(e[0], e[1], weight)

        # Run Infomap
        im.run()

        # Convert to dictionary format, ensuring all nodes are included
        communities = {}
        
        # First pass: collect all module assignments
        for node in im.tree:
            if node.is_leaf:
                communities[node.physical_id] = node.module_id

        # Second pass: ensure all graph nodes have a community
        # If a node wasn't assigned by Infomap, give it its own community
        max_community = max(communities.values(), default=-1)
        for node in G.nodes():
            if node not in communities:
                max_community += 1
                communities[node] = max_community

        return communities
    except Exception as e:
        print(f"Error in Infomap algorithm: {e}")
        return detect_communities_networkx(G)


def detect_communities_fluid(G, k=None):
    """
    Detect communities using Fluid Communities algorithm.
    """
    if k is None:
        # Estimate k based on network size and connectivity
        k = max(2, min(int(np.sqrt(len(G.nodes()))), len(G.nodes()) - 1))

    try:
        communities_generator = nx.algorithms.community.asyn_fluidc(G, k, max_iter=100)
        communities_list = list(communities_generator)

        # Convert to dictionary format
        communities = {}
        for i, community in enumerate(communities_list):
            for node in community:
                communities[node] = i
        return communities
    except Exception as e:
        print(f"Error in Fluid Communities algorithm: {e}")
        return detect_communities_networkx(G)


def detect_communities_async_label_prop(G):
    """
    Detect communities using Asynchronous Label Propagation.
    """
    try:
        communities_list = list(
            nx.algorithms.community.asyn_lpa_communities(G, weight="weight")
        )
        communities = {}
        for i, community in enumerate(communities_list):
            for node in community:
                communities[node] = i
        return communities
    except Exception as e:
        print(f"Error in Async Label Propagation: {e}")
        return detect_communities_networkx(G)


def create_layer_communities_chart(
    heatmap_ax,
    network_ax,
    layer_connections,
    layers,
    medium_font,
    large_font,
    visible_layers=None,
    layer_colors=None,
    algorithm="Louvain",
):
    """
    Create visualizations of layer communities.

    Parameters:
    -----------
    heatmap_ax : matplotlib.axes.Axes
        Axes for the community heatmap visualization
    network_ax : matplotlib.axes.Axes
        Axes for the community network visualization
    layer_connections : numpy.ndarray
        Matrix of connection counts between layers (already filtered)
    layers : list
        List of visible layer names (already filtered)
    medium_font, large_font : dict
        Font configuration dictionaries
    visible_layers : list, optional
        Sequential indices for visible layers (0 to len(layers)-1)
    layer_colors : dict, optional
        Dictionary mapping layer names to colors
    algorithm : str
        The community detection algorithm to use
        ("Louvain", "Leiden", "Label Propagation", "Spectral", "Infomap", "Fluid", "AsyncLPA")
    """
    # No need to filter the layer_connections matrix as it's already filtered
    # Just check if we have any data to visualize
    if len(layers) > 0 and np.sum(layer_connections) > 0:
        # Create a graph where nodes are layers and edges represent connections
        G = nx.Graph()

        # Add nodes (layers) using sequential indices
        for i, layer in enumerate(layers):
            G.add_node(i, name=layer)

        # Add edges with weights based on connection counts
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                if layer_connections[i, j] > 0:
                    G.add_edge(i, j, weight=layer_connections[i, j])

        # Check if the graph has any edges
        if len(G.edges()) == 0:
            # No edges in the graph, display a message
            message = "No connections between visible layers"
            heatmap_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
            heatmap_ax.axis("off")

            network_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
            network_ax.axis("off")
            return

        # Detect communities based on selected algorithm
        if algorithm == "Louvain":
            # Use Louvain algorithm if available, otherwise use networkx's community detection
            if HAS_COMMUNITY:
                try:
                    communities = best_partition(G, weight="weight")
                    algorithm_name = "Louvain"
                except Exception as e:
                    # Fallback to networkx's community detection if python-louvain fails
                    communities = detect_communities_networkx(G)
                    algorithm_name = "Louvain (fallback)"
                    print(f"Error with Louvain algorithm: {e}")
            else:
                communities = detect_communities_networkx(G)
                algorithm_name = "Louvain (fallback)"
        elif algorithm == "Leiden":
            # Use Leiden algorithm if available, otherwise use networkx's community detection
            if HAS_LEIDEN:
                try:
                    # Convert networkx graph to igraph
                    edges = list(G.edges(data=True))
                    weights = [e[2].get("weight", 1.0) for e in edges]
                    ig_graph = ig.Graph(
                        n=len(G.nodes()),
                        edges=[(e[0], e[1]) for e in edges],
                        edge_attrs={"weight": weights},
                    )

                    # Run Leiden algorithm
                    partition = la.find_partition(
                        ig_graph, la.ModularityVertexPartition, weights="weight"
                    )

                    # Convert partition to the expected format
                    communities = {
                        node: membership
                        for node, membership in enumerate(partition.membership)
                    }
                    algorithm_name = "Leiden"
                except Exception as e:
                    communities = detect_communities_networkx(G)
                    algorithm_name = "Leiden (fallback)"
                    print(f"Error with Leiden algorithm: {e}")
            else:
                communities = detect_communities_networkx(G)
                algorithm_name = "Leiden (fallback)"
        elif algorithm == "Spectral":
            try:
                communities = detect_communities_spectral(G)
                algorithm_name = "Spectral"
            except Exception as e:
                communities = detect_communities_networkx(G)
                algorithm_name = "Spectral (fallback)"
                print(f"Error with Spectral Clustering: {e}")
        elif algorithm == "Infomap":
            try:
                communities = detect_communities_infomap(G)
                algorithm_name = "Infomap"
            except Exception as e:
                communities = detect_communities_networkx(G)
                algorithm_name = "Infomap (fallback)"
                print(f"Error with Infomap: {e}")
        elif algorithm == "Fluid":
            try:
                communities = detect_communities_fluid(G)
                algorithm_name = "Fluid"
            except Exception as e:
                communities = detect_communities_networkx(G)
                algorithm_name = "Fluid (fallback)"
                print(f"Error with Fluid Communities: {e}")
        elif algorithm == "AsyncLPA":
            try:
                communities = detect_communities_async_label_prop(G)
                algorithm_name = "Async Label Propagation"
            except Exception as e:
                communities = detect_communities_networkx(G)
                algorithm_name = "AsyncLPA (fallback)"
                print(f"Error with Async Label Propagation: {e}")
        else:  # Label Propagation
            # Use Label Propagation algorithm
            try:
                communities_list = list(
                    nx.algorithms.community.label_propagation_communities(G)
                )
                communities = {}
                for i, community in enumerate(communities_list):
                    for node in community:
                        communities[node] = i
                algorithm_name = "Label Propagation"
            except Exception as e:
                communities = detect_communities_networkx(G)
                algorithm_name = "Label Propagation (fallback)"
                print(f"Error with Label Propagation algorithm: {e}")

        # Get unique community IDs and count
        unique_communities = sorted(set(communities.values()))
        num_communities = len(unique_communities)

        # Create a colormap for communities - ensure we only generate colors for actual communities
        community_colors = plt.cm.tab10(np.linspace(0, 1, num_communities))

        # Map nodes to their community colors
        node_colors = []
        for node in G.nodes():
            comm_idx = unique_communities.index(communities[node])
            node_colors.append(community_colors[comm_idx])

        # Create a mapping of community ID to list of layers in that community
        community_members = {comm: [] for comm in unique_communities}
        for node, comm in communities.items():
            community_members[comm].append(layers[node])

        # Calculate community metrics
        try:
            modularity = nx.algorithms.community.modularity(
                G,
                [
                    [node for node in G.nodes() if communities[node] == comm]
                    for comm in unique_communities
                ],
            )
        except ZeroDivisionError:
            # Handle the case where modularity calculation fails
            modularity = 0.0
            print("Warning: Modularity calculation failed (division by zero)")
        except Exception as e:
            # Handle any other exceptions
            modularity = 0.0
            print(f"Warning: Modularity calculation failed: {e}")

        # Calculate bridge nodes (nodes with connections to multiple communities)
        bridge_nodes = []
        for node in G.nodes():
            node_comm = communities[node]
            neighbor_comms = set()
            for neighbor in G.neighbors(node):
                neighbor_comm = communities[neighbor]
                if neighbor_comm != node_comm:
                    neighbor_comms.add(neighbor_comm)
            if len(neighbor_comms) > 0:
                bridge_nodes.append(node)

        # Create community matrix visualization for the heatmap
        # Reorder the connection matrix by community
        community_order = []
        for comm in unique_communities:
            community_order.extend(
                [i for i, node in enumerate(G.nodes()) if communities[node] == comm]
            )

        reordered_connections = layer_connections[
            np.ix_(community_order, community_order)
        ]
        reordered_layers = [layers[i] for i in community_order]

        # Create a matrix plot
        im = heatmap_ax.imshow(reordered_connections, cmap="viridis")

        # Add colorbar
        cbar = heatmap_ax.figure.colorbar(im, ax=heatmap_ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Connection Strength", fontsize=8)

        # Add community dividers
        current_pos = 0
        for comm in unique_communities:
            comm_size = sum(1 for node in G.nodes() if communities[node] == comm)
            if current_pos > 0:
                heatmap_ax.axhline(
                    y=current_pos - 0.5, color="white", linestyle="-", linewidth=2
                )
                heatmap_ax.axvline(
                    x=current_pos - 0.5, color="white", linestyle="-", linewidth=2
                )
            current_pos += comm_size

        # Add labels
        heatmap_ax.set_xticks(range(len(reordered_layers)))
        heatmap_ax.set_yticks(range(len(reordered_layers)))
        heatmap_ax.set_xticklabels(reordered_layers, rotation=90, fontsize=6)
        heatmap_ax.set_yticklabels(reordered_layers, fontsize=6)

        # Add community labels
        current_pos = 0
        for comm in unique_communities:
            comm_size = sum(1 for node in G.nodes() if communities[node] == comm)
            if comm_size > 0:
                heatmap_ax.text(
                    current_pos + comm_size / 2 - 0.5,
                    -1,
                    f"C{comm}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(
                        facecolor=community_colors[comm % len(community_colors)],
                        alpha=0.7,
                    ),
                )
                heatmap_ax.text(
                    -1,
                    current_pos + comm_size / 2 - 0.5,
                    f"C{comm}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(
                        facecolor=community_colors[comm % len(community_colors)],
                        alpha=0.7,
                    ),
                )
            current_pos += comm_size

        heatmap_ax.set_title(f"Community Connection Matrix", **large_font)

        # Add community legend
        legend_text = "Communities:\n" + "\n".join(
            [
                f"C{comm}: {', '.join(members[:3])}"
                + (f" + {len(members) - 3} more" if len(members) > 3 else "")
                for comm, members in community_members.items()
            ]
        )
        heatmap_ax.text(
            1.05,
            0.5,
            legend_text,
            transform=heatmap_ax.transAxes,
            ha="left",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Draw the community graph in the network_ax
        pos = nx.spring_layout(G, seed=42, weight="weight")

        # Draw nodes with community colors
        nx.draw_networkx_nodes(
            G, pos, node_size=300, node_color=node_colors, ax=network_ax
        )

        # Highlight bridge nodes with a black border
        if bridge_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=bridge_nodes,
                node_size=300,
                node_color=node_colors,
                edgecolors="black",
                linewidths=2,
                ax=network_ax,
            )

        # Draw edges with width proportional to weight
        edge_widths = [G[u][v]["weight"] / 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, ax=network_ax)

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            labels={
                i: f"{layer}\n(C{communities[i]})"
                for i, layer in enumerate(layers)
            },
            font_size=8,
            ax=network_ax,
        )

        network_ax.set_title(f"Layer Communities ({algorithm_name})", **large_font)
        network_ax.axis("off")

        # Add community information as text
        info_text = (
            f"Communities detected: {num_communities}\n"
            f"Modularity: {modularity:.3f}\n"
            f"Bridge layers: {len(bridge_nodes)}"
        )
        network_ax.text(
            0.5,
            -0.1,
            info_text,
            transform=network_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    else:
        if len(layers) == 0:
            message = "No visible layers to display"
        else:
            message = "No interlayer connections to display"

        heatmap_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
        heatmap_ax.axis("off")

        network_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
        network_ax.axis("off")


def detect_communities_networkx(G):
    """
    Fallback community detection using networkx's algorithms.

    Parameters:
    -----------
    G : networkx.Graph
        The graph to detect communities in

    Returns:
    --------
    dict
        A dictionary mapping node IDs to community IDs
    """
    # Check if the graph has any edges
    if len(G.edges()) == 0:
        # No edges, assign all nodes to the same community
        return {node: 0 for node in G.nodes()}

    # Try to use Girvan-Newman algorithm (divisive hierarchical clustering)
    try:
        # This is a hierarchical method, so we need to choose a level
        communities_generator = nx.algorithms.community.girvan_newman(G)
        # Take the first level with at least 2 communities
        for communities in communities_generator:
            if len(communities) >= 2:
                break

        # Convert to the expected format (node -> community ID)
        communities_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                communities_dict[node] = i

        return communities_dict
    except:
        # If Girvan-Newman fails, try greedy modularity maximization
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(
                G, weight="weight"
            )
            communities_dict = {}
            for i, community in enumerate(communities):
                for node in community:
                    communities_dict[node] = i

            return communities_dict
        except:
            # Last resort: assign all nodes to the same community
            print("XXXXX Warning: All nodes assigned to the same community, couldnt run algos")
            return {node: 0 for node in G.nodes()}
