import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class SpiralDiveAnimator(BaseAnimator):
    """Spiral dive animation that spirals down into the network"""

    def animate(self, duration=None):
        """
        Spiral dive animation - camera spirals down into the network.
        Creates a dramatic diving effect with continuous rotation.

        Args:
            duration (float, optional): Duration of the animation in seconds.
                                      If None, uses the default duration.
        """
        if self._animation_in_progress:
            return

        self._animation_in_progress = True
        current_state = self._store_camera_state()

        # Determine if we're in orthographic mode by checking fov
        is_orthographic = self.view.camera.fov == 0

        # Animation parameters
        params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": self._duration_to_frames(duration),
            "spiral_rotations": 2,  # Number of complete rotations
            "current_frame": 0,
            "restore_original": True,
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / self.FRAME_RATE,
            connect=lambda _: self._animation_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _animation_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        spiral_rotations,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the spiral dive animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Use elastic easing for a bouncy finish
        if progress < 0.65:
            segment_progress = progress / 0.65
            eased_progress = self.ease_out_elastic(segment_progress)
        else:
            # Last 15% - return to original position
            segment_progress = (progress - 0.45) / 0.55
            eased_progress = 1.0 - segment_progress  # Linear return to start

        # Calculate zoom factor
        # Start zoomed out (5x), end at current zoom level
        zoom_factor = 2 * (1 - eased_progress) + 1 * eased_progress

        # Calculate spiral rotation
        if progress < 0.35:
            # First 85% - complete the spiral
            rotation_progress = progress / 0.65
            azimuth_change = 360 * spiral_rotations * rotation_progress
        else:
            # Last 15% - return to original azimuth
            segment_progress = (progress - 0.65) / 0.35
            azimuth_change = 360 * spiral_rotations * (1 - segment_progress)

        # Calculate elevation change (start high, spiral down)
        start_elevation = 60  # Start looking from above (less extreme)

        if progress < 0.85:
            # First 85% - spiral down
            elevation_current = (
                start_elevation
                - (start_elevation - current_state["elevation"]) * eased_progress
            )
        else:
            # Last 15% - return to original elevation
            segment_progress = (progress - 0.85) / 0.15
            elevation_offset = (start_elevation - current_state["elevation"]) * (
                1 - segment_progress
            )
            elevation_current = current_state["elevation"] + elevation_offset

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            # For scale_factor, LARGER values zoom in, SMALLER values zoom out
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 / zoom_factor
            )
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            self.view.camera.distance = current_state["distance"] * zoom_factor

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = elevation_current

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "spiral_rotations": spiral_rotations,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
