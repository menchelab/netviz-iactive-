import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class OrbitFlybyAnimator(BaseAnimator):
    """Orbit flyby animation that flies around the network in an elliptical orbit"""

    def animate(self):
        """
        Orbit flyby animation - camera flies around the network in an elliptical orbit,
        changing elevation to create a dynamic flyby effect.
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
            "total_frames": 60,  # Reduced to 60 frames
            "orbit_cycles": 1.0,  # Exactly one full orbit
            "elevation_amplitude": 30,  # Maximum elevation change
            "distance_factor": 1.2,  # Increase distance slightly for better view
            "current_frame": 0,
            "restore_original": True,  # Return to original position at end
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
        orbit_cycles,
        elevation_amplitude,
        distance_factor,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the orbit flyby animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Calculate azimuth change (full orbits)
        # Use sine easing to start and end smoothly at the original position
        if progress < 0.7:  # Changed from 0.9 to 0.7 to give more time for ending
            # First 70% - complete the orbit
            orbit_progress = progress / 0.7
            # Add ease-out to make it smoother
            eased_orbit = self.ease_out_cubic(orbit_progress)
            azimuth_change = 360 * orbit_cycles * eased_orbit
        else:
            # Last 30% - return to original azimuth more gradually
            final_segment = (progress - 0.7) / 0.3
            # Use ease-in-out for smoother return
            eased_final = self.ease_in_out_cubic(final_segment)
            azimuth_change = 360 * orbit_cycles * (1 - eased_final)

        # Calculate elevation using a sine wave for up-down motion
        # Ensure we start and end at 0 elevation change
        elevation_offset = elevation_amplitude * np.sin(progress * np.pi * 2)

        # Ensure elevation returns to original more gradually
        if progress > 0.7:
            final_segment = (progress - 0.7) / 0.3
            eased_final = self.ease_in_out_cubic(final_segment)
            elevation_offset = elevation_offset * (1 - eased_final)

        # Calculate distance/scale variation (closer and further) using a sine wave
        variation_factor = 0.2 * np.sin(progress * np.pi * 3)

        # Ensure variation returns to 0 more gradually
        if progress > 0.7:
            final_segment = (progress - 0.7) / 0.3
            eased_final = self.ease_in_out_cubic(final_segment)
            variation_factor = variation_factor * (1 - eased_final)

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            # For scale_factor, LARGER values zoom in, SMALLER values zoom out
            zoom_factor = distance_factor + variation_factor

            # Ensure we return to original scale gradually
            if progress > 0.7:
                final_segment = (progress - 0.7) / 0.3
                eased_final = self.ease_in_out_cubic(final_segment)
                zoom_factor = zoom_factor * (1 - eased_final) + 1 * eased_final

            self.view.camera.scale_factor = current_state["scale_factor"] / zoom_factor
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            distance_change = (
                current_state["distance"] * distance_factor - current_state["distance"]
            ) + (current_state["distance"] * variation_factor)

            # Ensure we return to original distance gradually
            if progress > 0.7:
                final_segment = (progress - 0.7) / 0.3
                eased_final = self.ease_in_out_cubic(final_segment)
                distance_change = distance_change * (1 - eased_final)

            self.view.camera.distance = current_state["distance"] + distance_change

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = current_state["elevation"] + elevation_offset

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "orbit_cycles": orbit_cycles,
            "elevation_amplitude": elevation_amplitude,
            "distance_factor": distance_factor,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
