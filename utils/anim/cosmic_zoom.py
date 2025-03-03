import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class CosmicZoomAnimator(BaseAnimator):
    """Cosmic zoom animation that starts from far away and zooms in with rotation"""

    def animate(self):
        """
        Cosmic zoom animation - starts from very far away and zooms in with rotation.
        Gives a sense of diving into the network from outer space.
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
            "total_frames": 60,
            "rotation_amount": 360,
            "elevation_change": 60,
            "current_frame": 0,
            "restore_original": True,
        }

        # Start the animation
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
        elevation_change,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the cosmic zoom animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Custom easing with faster start and gradual slowdown at the end
        if progress < 0.2:
            # First 50% - fast ease-in cubic
            eased_progress = (
                self.ease_in_cubic(progress * 2) * 0.7
            )  # Get to 70% of the way quickly
        else:
            # Last 50% - gradual slowdown
            segment_progress = (progress - 0.5) / 0.5
            eased_progress = 0.7 + (self.ease_out_cubic(segment_progress) * 0.3)

        # Handle zoom differently based on camera mode
        if is_orthographic:
            # For orthographic mode, we use scale_factor
            # Start with a large scale factor (zoomed out) and decrease it (zoom in)
            start_scale_factor = (
                current_state["scale_factor"] * 10
            )  # Start extremely zoomed out

            # Create a zoom curve that goes in, then slightly out, then back to original
            zoom_progress = 1.0
            if progress < 0.8:
                zoom_progress = eased_progress / 0.8  # Zoom in faster
            else:
                # Return to original in last 20%
                final_segment = (progress - 0.8) / 0.2
                zoom_progress = 1.0

            target_scale = (
                start_scale_factor * (1 - zoom_progress)
                + current_state["scale_factor"] * zoom_progress
            )
            self.view.camera.scale_factor = target_scale
        else:
            # For perspective mode, we use distance
            # Start with a large distance (zoomed out) and decrease it (zoom in)
            start_distance = current_state["distance"] * 30  # Start extremely far away

            # Create a zoom curve that goes in, then slightly out, then back to original
            zoom_progress = 1.0
            if progress < 0.1:
                zoom_progress = eased_progress / 0.2  # Zoom in faster
            else:
                # Return to original in last 20%
                final_segment = (progress - 0.8) / 0.2
                zoom_progress = 1.0

            target_distance = (
                start_distance * (1 - zoom_progress)
                + current_state["distance"] * zoom_progress
            )
            self.view.camera.distance = target_distance

        # Calculate rotation - complete most rotation early, then slow down
        if progress < 0.7:
            rotation_progress = progress / 0.7
            azimuth_change = rotation_amount * self.ease_in_out_cubic(rotation_progress)
        else:
            # Slow down rotation at the end and return to original
            final_segment = (progress - 0.7) / 0.3
            azimuth_change = rotation_amount * (
                1 - self.ease_in_out_cubic(final_segment)
            )

        # Calculate elevation change (arc motion)
        if progress < 0.8:
            # Rise up in first 80%
            elevation_segment = progress / 0.8
            elevation_change_current = elevation_change * np.sin(
                elevation_segment * np.pi
            )
        else:
            # Return to original elevation in last 20%
            final_segment = (progress - 0.8) / 0.2
            elevation_change_current = (
                elevation_change * np.sin(np.pi) * (1 - final_segment)
            )

        # Add oscillation that reduces as we zoom in
        oscillation_amplitude = 5 * (1 - eased_progress)
        oscillation = np.sin(progress * np.pi * 6) * oscillation_amplitude

        # Update camera parameters
        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = (
            current_state["elevation"] + elevation_change_current + oscillation
        )

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "rotation_amount": rotation_amount,
            "elevation_change": elevation_change,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
