import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line

class CanvasManager:
    """Manages the 3D visualization canvas for the network"""
    
    def __init__(self):
        # Create canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600)) 