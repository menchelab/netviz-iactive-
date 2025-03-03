import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class BounceZoomAnimator(BaseAnimator):
    """Bounce zoom animation that zooms in with a bouncy effect"""

    def animate(self):
        """
        Bounce zoom animation - camera zooms in with a bouncy effect,
        as if it's bouncing off an elastic surface.
        """
        if self._animation_in_progress:
            return

        self._animation_in_progress = True
        current_state = self._store_camera_state()

        is_orthographic = self.view.camera.fov == 0

        params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": 60,
            "rotation_amount": 360,
            "current_frame": 0,
            "restore_original": True,
        }

        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._animation_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _animation_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        rotation_amount,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the bounce zoom animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Use bounce easing for a bouncy finish
        if progress < 0.85:
            # First 85% - bounce zoom with bounce easing
            segment_progress = progress / 0.85
            # Reduce frequency of bounce by adjusting the progress
            bounce_progress = segment_progress * 0.7  # Reduce bounce frequency
            eased_progress = self.ease_out_bounce(bounce_progress)
            # Reduce the zoom range to be less extreme
            zoom_factor = (
                2.5 * (1 - eased_progress) + 1 * eased_progress
            )  # Reduced from 3.5 to 2.5
        else:
            # Last 15% - return to original position smoothly
            segment_progress = (progress - 0.85) / 0.15
            eased_progress = self.ease_in_out_cubic(segment_progress)
            zoom_factor = 1  # Stay at original zoom level during return

        rotation_progress = progress
        eased_rotation = self.ease_in_out_cubic(rotation_progress)
        azimuth_change = rotation_amount * eased_rotation

        # Add some bounce to elevation with reduced amplitude
        if progress < 0.85:
            # First 85% - bounce elevation
            elevation_bounce = 10 * np.sin(progress * np.pi * 3)  # Reduced amplitude
        else:
            # Last 15% - return to original elevation smoothly
            segment_progress = (progress - 0.85) / 0.15
            eased_progress = self.ease_in_out_cubic(segment_progress)
            elevation_bounce = 10 * np.sin(progress * np.pi * 3) * (1 - eased_progress)

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
        self.view.camera.elevation = current_state["elevation"] + elevation_bounce

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "rotation_amount": rotation_amount,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
