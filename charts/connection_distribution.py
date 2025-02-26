import collections
import numpy as np

def create_connection_distribution_chart(ax, intralayer_connections, small_font, medium_font):
    """Create bar chart showing distribution of connection counts"""
    if intralayer_connections:
        # Count how many nodes have X connections
        conn_dist = collections.Counter(intralayer_connections.values())
        # Sort by connection count
        conn_counts = sorted(conn_dist.items())
        x, y = zip(*conn_counts) if conn_counts else ([], [])
        
        ax.bar(x, y, width=0.7)
        ax.set_xlabel('Number of Connections', **small_font)
        ax.set_ylabel('Number of Nodes', **small_font)
        ax.set_title('Intra-Connection Distribution', **medium_font)
        ax.tick_params(axis='both', labelsize=6)
    else:
        ax.text(0.5, 0.5, 'No connection data to display', 
               horizontalalignment='center', verticalalignment='center', **small_font) 