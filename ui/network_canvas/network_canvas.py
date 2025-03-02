import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text
import logging
import time
from vispy import app

from .node_manager import NodeManager
from .edge_manager import EdgeManager
from .label_manager import LabelManager
from .visibility_manager import VisibilityManager
from utils.anim.animation_manager import AnimationManager

logger = logging.getLogger(__name__)

class NetworkCanvas:
    def __init__(self, parent=None, data_manager=None):
        logger.info("Creating canvas...")

        # Store reference to data manager
        self.data_manager = data_manager
        if not data_manager:
            logger.error("NetworkCanvas requires a data_manager")
            raise ValueError("NetworkCanvas requires a data_manager")

        # Create canvas and view
        self.canvas = scene.SceneCanvas(keys="interactive", size=(1200, 768))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.fov = 45
        self.view.camera.distance = 7  # for non orthographic view
        self.view.camera.scale_factor = 4  # for orthographic view

        # Position camera higher on z-axis
        initial_center = self.view.camera.center
        self.view.camera.center = np.array(
            [initial_center[0], initial_center[1], initial_center[2] + 1.5]
        )

        # Set up key event handling
        self.canvas.events.key_press.connect(self._on_key_press)
        # Store initial camera state for reset
        self._initial_camera_state = {
            "center": self.view.camera.center,
            "distance": self.view.camera.distance,
            "elevation": self.view.camera.elevation,
            "azimuth": self.view.camera.azimuth,
            "scale_factor": self.view.camera.scale_factor,
        }

        # Initialize animation manager
        self.animation_manager = AnimationManager(self.view)

        # Create visuals
        self.scatter = Markers()
        self.view.add(self.scatter)

        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=1
        )
        self.view.add(self.intralayer_lines)

        self.interlayer_lines = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=1
        )
        self.view.add(self.interlayer_lines)

        # Create text labels for nodes
        self.node_labels = Text(
            pos=np.array([[0, 0, 0]]), text=[""], color="white", font_size=6
        )
        self.node_labels.visible = False
        self.view.add(self.node_labels)

        # Create line visuals for bar charts
        self.node_count_bars = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=3
        )
        self.node_count_bars.visible = False
        self.view.add(self.node_count_bars)

        self.edge_count_bars = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=3
        )
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

    def _on_key_press(self, event):
        """Handle key press events for navigation and view control"""

        move_amount = 0.5
        if event.key == "c":
            self._center_view()
        elif event.key == "r":
            self._reset_camera_and_rotation()
        # Camera animation keys
        elif event.key == " ":  # Spacebar - Cosmic Zoom
            self.animation_manager.play_animation(AnimationManager.COSMIC_ZOOM)
        elif event.key == "u":  # Orbit Flyby
            self.animation_manager.play_animation(AnimationManager.ORBIT_FLYBY)
        elif event.key == "i":  # Spiral Dive
            self.animation_manager.play_animation(AnimationManager.SPIRAL_DIVE)
        elif event.key == "o":  # Bounce Zoom
            self.animation_manager.play_animation(AnimationManager.BOUNCE_ZOOM)
        elif event.key == "j":  # Swing Around
            self.animation_manager.play_animation(AnimationManager.SWING_AROUND)
        elif event.key == "k":  # Pulse Zoom
            self.animation_manager.play_animation(AnimationManager.PULSE_ZOOM)
        elif event.key == "l":  # Matrix Effect
            self.animation_manager.play_animation(AnimationManager.MATRIX_EFFECT)
        elif event.key == "m":  # Slow Matrix Effect
            self.animation_manager.play_animation(AnimationManager.MATRIX_EFFECT_SLOW)
        # Arrow keys and WASD for rotation
        elif event.key in ("Left", "a"):
            self._rotate_z(-move_amount)  # Rotate left around z-axis
        elif event.key in ("Right", "d"):
            self._rotate_z(move_amount)  # Rotate right around z-axis
        elif event.key in ("Up", "w"):
            self._rotate_x(move_amount)  # Rotate up around x-axis
        elif event.key in ("Down", "s"):
            self._rotate_x(-move_amount)  # Rotate down around x-axis
        # Q and E for y-axis rotation
        elif event.key == "q":
            self._rotate_y(-move_amount)  # Rotate left around y-axis
        elif event.key == "e":
            self._rotate_y(move_amount)  # Rotate right around y-axis
        # Axis view keys
        elif event.key == "x":
            self._look_along_axis("x")
        elif event.key == "y":
            self._look_along_axis("y")
        elif event.key == "z":
            self._look_along_axis("z")

    def _center_view(self):
        self.view.camera.center = self._initial_camera_state["center"]
        self.view.camera.distance = self._initial_camera_state["distance"]
        self.canvas.update()

    def _reset_camera_and_rotation(self):
        """Reset camera to an isometric diagonal view"""
        # Reset to initial camera state for position and distance
        self.view.camera.center = self._initial_camera_state["center"]
        self.view.camera.distance = self._initial_camera_state["distance"]
        self.view.camera.scale_factor = self._initial_camera_state["scale_factor"]

        # Set to isometric view angles
        # Azimuth: 45 degrees (halfway between x and y axes)
        # Elevation: 35.264 degrees (approximately arctan(1/sqrt(2)), the isometric angle)
        self.view.camera.azimuth = 45
        self.view.camera.elevation = 45

        # Reset roll if available
        if hasattr(self.view.camera, "roll"):
            self.view.camera.roll = 0

        self.canvas.update()

    def _reset_camera(self):
        """Reset camera to initial state"""
        self.view.camera.center = self._initial_camera_state["center"]
        self.view.camera.distance = self._initial_camera_state["distance"]
        self.view.camera.elevation = self._initial_camera_state["elevation"]
        self.view.camera.azimuth = self._initial_camera_state["azimuth"]

    def _rotate_z(self, degrees):
        """Rotate the view around the z-axis"""
        self.view.camera.azimuth += degrees
        self.canvas.update()

    def _rotate_x(self, degrees):
        """Rotate the view around the x-axis"""
        self.view.camera.elevation += degrees
        self.canvas.update()

    def _rotate_y(self, degrees):
        """Rotate the view around the y-axis (roll)"""
        if hasattr(self.view.camera, "roll"):
            self.view.camera.roll += degrees
            self.canvas.update()

    def _look_along_axis(self, axis):
        """Set the camera to look along the specified axis toward the center"""
        # Store current center and distance
        center = self.view.camera.center
        distance = self.view.camera.distance

        # Reset camera orientation
        if axis == "x":
            # Look along x-axis (from positive x)
            self.view.camera.azimuth = 0
            self.view.camera.elevation = 0
        elif axis == "y":
            # Look along y-axis (from positive y)
            self.view.camera.azimuth = 90
            self.view.camera.elevation = 0
        elif axis == "z":
            # Look along z-axis (from top)
            self.view.camera.azimuth = 0
            self.view.camera.elevation = 90

        # Restore center and distance
        self.view.camera.center = center
        self.view.camera.distance = distance

        # Update the view
        self.canvas.update()

    def load_data(
        self,
        node_positions=None,
        link_pairs=None,
        link_colors=None,
        node_colors=None,
        node_ids=None,
    ):
        return self.node_manager.load_data(
            node_positions=node_positions,
            link_pairs=link_pairs,
            link_colors=link_colors,
            node_colors=node_colors,
            node_ids=node_ids,
        )

    def set_layer_colors(self, layer_colors):
        """Set the layer colors mapping"""
        return self.node_manager.set_layer_colors(layer_colors)

    def update_visibility(
        self,
        node_mask=None,
        edge_mask=None,
        show_intralayer=True,
        show_nodes=True,
        show_labels=True,
        bottom_labels_only=True,
        show_stats_bars=False,
    ):
        """Update the visibility of nodes and edges based on masks"""
        return self.visibility_manager.update_visibility(
            node_mask=node_mask,
            edge_mask=edge_mask,
            show_intralayer=show_intralayer,
            show_nodes=show_nodes,
            show_labels=show_labels,
            bottom_labels_only=bottom_labels_only,
            show_stats_bars=show_stats_bars,
        )

    def set_projection_mode(self, orthographic=True):
        if orthographic:
            self.view.camera.fov = 0
        else:
            self.view.camera.fov = 45

        self.canvas.update()

    def update_with_optimized_data(
        self,
        optimized_data,
        show_intralayer=True,
        show_nodes=True,
        show_labels=True,
        bottom_labels_only=True,
        show_stats_bars=False,
    ):
        """
        Update the visualization with optimized data from the NetworkDataManager.
        This method is designed for efficient rendering of large networks.
        
        Parameters:
        -----------
        optimized_data : dict
            Dictionary containing optimized data from NetworkDataManager.optimize_for_vispy():
            - 'node_positions': NumPy array of node positions
            - 'node_colors': NumPy array of node colors (RGBA)
            - 'node_sizes': NumPy array of node sizes
            - 'edge_connections': NumPy array of edge connections
            - 'edge_colors': NumPy array of edge colors (RGBA)
            - 'edge_importance': NumPy array of edge importance scores (for LOD)
            - 'is_simplified': Boolean indicating if the data was simplified
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input data
            if optimized_data is None:
                logger.error("No optimized data provided")
                return
                
            # Check for required keys
            required_keys = ['node_positions', 'node_colors', 'node_sizes', 'edge_connections', 'edge_colors']
            missing_keys = [key for key in required_keys if key not in optimized_data]
            if missing_keys:
                logger.error(f"Missing required keys in optimized data: {missing_keys}")
                return
            
            # Extract data from the optimized data dictionary
            node_positions = optimized_data['node_positions']
            node_colors = optimized_data['node_colors']
            node_sizes = optimized_data['node_sizes']
            edge_connections = optimized_data['edge_connections']
            edge_colors = optimized_data['edge_colors']
            edge_importance = optimized_data.get('edge_importance', None)
            is_simplified = optimized_data.get('is_simplified', False)
            
            if is_simplified:
                logger.info(f"Rendering simplified network: {len(node_positions)} nodes, {len(edge_connections)} edges")
            
            # Update node manager with the optimized data
            self.node_manager.update_with_optimized_data(
                node_positions=node_positions,
                node_colors=node_colors,
                node_sizes=node_sizes,
                visible=show_nodes
            )
            
            # Update edge manager with the optimized data
            self.edge_manager.update_with_optimized_data(
                edge_connections=edge_connections,
                edge_colors=edge_colors,
                edge_importance=edge_importance,
                show_intralayer=show_intralayer
            )
            
            # Update label manager
            if show_labels:
                # For labels, we need to use the original node IDs
                # We can get these from the data manager
                if hasattr(self.data_manager, 'node_ids') and self.data_manager.node_ids is not None:
                    # Get visible node indices
                    if hasattr(self.data_manager, 'current_node_mask') and self.data_manager.current_node_mask is not None:
                        visible_indices = np.where(self.data_manager.current_node_mask)[0]
                        visible_node_ids = [self.data_manager.node_ids[i] for i in visible_indices if i < len(self.data_manager.node_ids)]
                        self.label_manager.update_labels(
                            node_positions=node_positions,
                            node_ids=visible_node_ids,
                            bottom_only=bottom_labels_only
                        )
            else:
                self.label_manager.hide_labels()
            
            # Update stats bars if needed
            if show_stats_bars and hasattr(self.data_manager, 'get_interlayer_edge_counts'):
                try:
                    interlayer_edge_counts = self.data_manager.get_interlayer_edge_counts()
                    if hasattr(self, 'stats_manager'):
                        self.stats_manager.update_stats_bars(
                            node_positions=node_positions,
                            interlayer_edge_counts=interlayer_edge_counts
                        )
                except Exception as e:
                    logger.error(f"Error updating stats bars: {e}")
                    if hasattr(self, 'stats_manager'):
                        self.stats_manager.hide_stats_bars()
            else:
                if hasattr(self, 'stats_manager'):
                    self.stats_manager.hide_stats_bars()
            
            # Update the canvas
            self.canvas.update()
            
            logger.debug(f"Updated visualization with optimized data: {len(node_positions)} nodes, {len(edge_connections)} edges")
        except Exception as e:
            logger.error(f"Error updating visualization with optimized data: {e}")
            import traceback
            traceback.print_exc()
