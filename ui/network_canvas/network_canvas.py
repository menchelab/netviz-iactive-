import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text
import logging

from .node_manager import NodeManager
from .edge_manager import EdgeManager
from .label_manager import LabelManager
from .visibility_manager import VisibilityManager

class NetworkCanvas:
    def __init__(self, parent=None, data_manager=None):
        logger = logging.getLogger(__name__)
        logger.info("Creating canvas...")

        # Store reference to data manager
        self.data_manager = data_manager
        if not data_manager:
            logger.error("NetworkCanvas requires a data_manager")
            raise ValueError("NetworkCanvas requires a data_manager")

        # Create canvas and view
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 768))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 3

        # Create visuals
        self.scatter = Markers()
        self.view.add(self.scatter)

        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=1)
        self.view.add(self.intralayer_lines)

        self.interlayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=2)
        self.view.add(self.interlayer_lines)
        
        # Create text labels for nodes
        self.node_labels = Text(pos=np.array([[0, 0, 0]]), text=[''], color='white', font_size=6)
        self.node_labels.visible = False
        self.view.add(self.node_labels)
        
        # Create line visuals for bar charts
        self.node_count_bars = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                  connect='segments', width=3)
        self.node_count_bars.visible = False
        self.view.add(self.node_count_bars)
        
        self.edge_count_bars = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                  connect='segments', width=3)
        self.edge_count_bars.visible = False
        self.view.add(self.edge_count_bars)
        
        # Store current visibility state
        self.current_node_mask = None
        self.current_edge_mask = None
        self.visible_layers = None
        
        # Initialize managers
        self.node_manager = NodeManager(self)
        self.edge_manager = EdgeManager(self)
        self.label_manager = LabelManager(self)
        self.visibility_manager = VisibilityManager(self)

    def load_data(self, node_positions=None, link_pairs=None, link_colors=None, node_colors=None, node_ids=None):
        """Load data either directly or from the data manager"""
        return self.node_manager.load_data(
            node_positions=node_positions,
            link_pairs=link_pairs,
            link_colors=link_colors,
            node_colors=node_colors,
            node_ids=node_ids
        )

    def set_layer_colors(self, layer_colors):
        """Set the layer colors mapping"""
        return self.node_manager.set_layer_colors(layer_colors)

    def update_visibility(self, node_mask=None, edge_mask=None, show_intralayer=True, 
                         show_nodes=True, show_labels=True, bottom_labels_only=True, 
                         show_stats_bars=False):
        """Update the visibility of nodes and edges based on masks"""
        return self.visibility_manager.update_visibility(
            node_mask=node_mask,
            edge_mask=edge_mask,
            show_intralayer=show_intralayer,
            show_nodes=show_nodes,
            show_labels=show_labels,
            bottom_labels_only=bottom_labels_only,
            show_stats_bars=show_stats_bars
        ) 