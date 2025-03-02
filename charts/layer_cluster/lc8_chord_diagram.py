import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
import matplotlib.patches as patches

def create_layer_cluster_chord(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,  # This is now an integer
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    cluster_colors=None,
):
    """
    Create a chord diagram showing relationships between layers and clusters.
    Arcs represent the number of nodes in each layer-cluster combination.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the chord diagram on
    visible_links : list of tuples
        List of (start_idx, end_idx) tuples representing visible links
    node_ids : list
        List of node IDs
    node_clusters : dict
        Dictionary mapping node IDs to cluster labels
    nodes_per_layer : int
        Number of nodes in each layer
    layers : list
        List of layer names
    small_font : dict
        Dictionary with font properties for small text
    medium_font : dict
        Dictionary with font properties for medium text
    visible_layer_indices : list, optional
        List of indices of visible layers
    cluster_colors : dict, optional
        Dictionary mapping cluster labels to colors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer-cluster chord diagram with nodes_per_layer={nodes_per_layer}")
    logger.info(f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Clusters: {len(node_clusters)}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Visible layer indices: {visible_layer_indices}")
    
    # Handle font parameters correctly
    if isinstance(medium_font, dict):
        medium_fontsize = medium_font.get("fontsize", 12)
    else:
        medium_fontsize = medium_font
        
    if isinstance(small_font, dict):
        small_fontsize = small_font.get("fontsize", 10)
    else:
        small_fontsize = small_font
    
    logger.info(f"Using medium_fontsize={medium_fontsize}, small_fontsize={small_fontsize}")

    # Clear the axis
    ax.clear()
    ax.set_title("Layer-Cluster Chord Diagram", fontsize=medium_fontsize)
    
    # Set equal aspect ratio instead of polar projection
    ax.set_aspect('equal')

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering chord diagram to show only {len(visible_layers)} visible layers")
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Get visible node indices
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Count nodes by cluster and layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    # Track all unique clusters and layers
    unique_clusters = set()
    
    # Process each visible node
    for node_idx in visible_node_indices:
        # Calculate layer index using integer division
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        # Add to counts
        cluster_layer_counts[cluster][layer_idx] += 1
        unique_clusters.add(cluster)
    
    logger.info(f"Counted nodes by cluster and layer: {dict(cluster_layer_counts)}")
    
    # Get unique layers (sorted)
    unique_layer_indices = sorted(visible_layers)
    unique_layers = [layers[i] for i in unique_layer_indices]
    
    # Get unique clusters (sorted)
    unique_clusters = sorted(unique_clusters)
    
    logger.info(f"Found {len(unique_clusters)} unique clusters across {len(unique_layers)} layers")
    
    # Check if we have any data
    if not unique_clusters or not unique_layers:
        logger.warning("No layer-cluster data to display")
        ax.text(
            0.5,
            0.5,
            "No layer-cluster data to display",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return
    
    try:
        # Create a list of all entities (layers and clusters)
        all_entities = unique_layers + unique_clusters
        num_entities = len(all_entities)
        
        logger.info(f"Total entities for chord diagram: {num_entities}")
        
        # Create a matrix of connections
        connection_matrix = np.zeros((num_entities, num_entities))
        
        # Fill the matrix with layer-cluster connections
        for i, layer_name in enumerate(unique_layers):
            layer_idx = unique_layer_indices[i]
            for j, cluster in enumerate(unique_clusters):
                cluster_idx = j + len(unique_layers)  # Offset by number of layers
                count = cluster_layer_counts[cluster][layer_idx]
                if count > 0:
                    connection_matrix[i, cluster_idx] = count
                    connection_matrix[cluster_idx, i] = count  # Make it symmetric
        
        logger.info(f"Created connection matrix of size {num_entities}x{num_entities}")
        
        # Calculate positions around the circle
        radius = 1.0
        angles = np.linspace(0, 2*np.pi, num_entities, endpoint=False)
        
        # Calculate entity positions
        entity_positions = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            entity_positions.append((x, y))
        
        # Create colors for entities
        entity_colors = []
        
        # Layer colors (blues)
        layer_cmap = plt.cm.Blues
        for i in range(len(unique_layers)):
            entity_colors.append(layer_cmap(0.3 + 0.7 * i / max(1, len(unique_layers) - 1)))
        
        # Cluster colors
        if cluster_colors:
            for cluster in unique_clusters:
                if cluster in cluster_colors:
                    entity_colors.append(cluster_colors[cluster])
                else:
                    entity_colors.append((0.5, 0.5, 0.5, 1.0))
        else:
            # Generate colors based on cluster names
            cluster_cmap = plt.cm.Oranges
            for i in range(len(unique_clusters)):
                entity_colors.append(cluster_cmap(0.3 + 0.7 * i / max(1, len(unique_clusters) - 1)))
        
        logger.info(f"Created {len(entity_colors)} colors for entities")
        
        # Draw arcs for each entity
        arc_width = 0.1
        for i, (x, y) in enumerate(entity_positions):
            # Draw arc
            arc = patches.Wedge(
                (0, 0), radius + arc_width/2,
                np.degrees(angles[i] - np.pi/num_entities),
                np.degrees(angles[i] + np.pi/num_entities),
                width=arc_width,
                color=entity_colors[i],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(arc)
            
            # Add label
            label_radius = radius + arc_width + 0.1
            label_x = label_radius * np.cos(angles[i])
            label_y = label_radius * np.sin(angles[i])
            
            # Adjust text alignment based on position
            ha = "left" if label_x > 0 else "right"
            va = "center"
            
            # Rotate text for better readability
            rotation = np.degrees(angles[i])
            if rotation > 90 and rotation < 270:
                rotation = rotation - 180
            
            # Add entity name
            entity_name = all_entities[i]
            ax.text(
                label_x, label_y,
                entity_name,
                ha=ha, va=va,
                rotation=rotation,
                fontsize=small_fontsize,
                fontweight='bold'
            )
        
        logger.info(f"Drew {num_entities} arcs for entities")
        
        # Draw chords between connected entities
        max_connection = np.max(connection_matrix) if np.max(connection_matrix) > 0 else 1
        min_chord_width = 0.01
        max_chord_width = 0.1
        
        # Count connections for logging
        connection_count = 0
        
        # Draw connections
        for i in range(num_entities):
            for j in range(i+1, num_entities):
                value = connection_matrix[i, j]
                if value > 0:
                    # Calculate chord width based on connection strength
                    chord_width = min_chord_width + (max_chord_width - min_chord_width) * (value / max_connection)
                    
                    # Get positions
                    start_angle = angles[i]
                    end_angle = angles[j]
                    
                    # Calculate control points for a curved path
                    # Use Bezier curve for smooth chord
                    start_x = radius * np.cos(start_angle)
                    start_y = radius * np.sin(start_angle)
                    end_x = radius * np.cos(end_angle)
                    end_y = radius * np.sin(end_angle)
                    
                    # Control points (pull toward center for curve)
                    control1_x = 0.5 * start_x
                    control1_y = 0.5 * start_y
                    control2_x = 0.5 * end_x
                    control2_y = 0.5 * end_y
                    
                    # Create path
                    path = Path([
                        (start_x, start_y),
                        (control1_x, control1_y),
                        (control2_x, control2_y),
                        (end_x, end_y)
                    ], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    
                    # Create patch
                    patch = patches.PathPatch(
                        path,
                        facecolor='none',
                        edgecolor='gray',
                        linewidth=chord_width * 20,  # Scale up for visibility
                        alpha=0.5
                    )
                    ax.add_patch(patch)
                    
                    # Add text for significant connections
                    if value > max_connection * 0.2:
                        # Position text at midpoint of the chord
                        mid_x = (start_x + end_x) / 2
                        mid_y = (start_y + end_y) / 2
                        
                        # Adjust position toward center for better placement
                        text_x = mid_x * 0.7
                        text_y = mid_y * 0.7
                        
                        ax.text(
                            text_x, text_y,
                            f"{int(value)}",
                            ha='center', va='center',
                            fontsize=small_fontsize - 1,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                        )
                    
                    connection_count += 1
        
        logger.info(f"Drew {connection_count} connections between entities")
        
        # Set axis limits with some padding
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a title with more information
        ax.set_title(f"Layer-Cluster Chord Diagram\n({len(unique_clusters)} clusters, {len(unique_layers)} layers)", 
                    fontsize=medium_fontsize)
        
        logger.info(f"Successfully created chord diagram with {len(unique_clusters)} clusters and {len(unique_layers)} layers")
        
    except Exception as e:
        logger.error(f"Error creating chord diagram: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating chord diagram: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 