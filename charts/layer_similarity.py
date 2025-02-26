import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

def create_layer_similarity_chart(ax, layer_connections, layers, small_font, medium_font):
    """Create dendrogram showing layer similarity based on connections"""
    if len(layers) > 1 and np.sum(layer_connections) > 0:
        # Create a similarity matrix between layers based on their connections
        # We'll use the layer_connections matrix as a measure of similarity
        similarity_matrix = layer_connections.copy()
        
        # Ensure diagonal is zero (no self-connections for distance calculation)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Convert to a distance matrix (higher similarity = lower distance)
        # Add a small epsilon to avoid division by zero
        max_connections = np.max(similarity_matrix)
        if max_connections > 0:
            # Normalize and invert to get distances (1 - normalized similarity)
            distance_matrix = 1 - (similarity_matrix / max_connections)
            
            # Ensure diagonal is exactly zero to satisfy scipy's requirements
            np.fill_diagonal(distance_matrix, 0)
            
            # Perform hierarchical clustering
            linkage_matrix = hierarchy.linkage(
                squareform(distance_matrix), 
                method='average'  # Use average linkage
            )
            
            # Plot dendrogram
            hierarchy.dendrogram(
                linkage_matrix,
                labels=layers,
                orientation='right',
                leaf_font_size=6,
                ax=ax
            )
            
            ax.set_title('Layer Similarity', **medium_font)
            ax.tick_params(axis='both', labelsize=6)
        else:
            ax.text(0.5, 0.5, 'No layer connections to analyze', 
                   horizontalalignment='center', verticalalignment='center', **small_font)
    else:
        ax.text(0.5, 0.5, 'Not enough layers for similarity analysis', 
               horizontalalignment='center', verticalalignment='center', **small_font) 