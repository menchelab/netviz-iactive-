import logging
from .cosmic_zoom import CosmicZoomAnimator
from .orbit_flyby import OrbitFlybyAnimator
from .spiral_dive import SpiralDiveAnimator
from .bounce_zoom import BounceZoomAnimator
from .swing_around import SwingAroundAnimator
from .pulse_zoom import PulseZoomAnimator
from .matrix_effect import MatrixEffectAnimator, MatrixEffectSlowAnimator
import random  # Add import for random

logger = logging.getLogger(__name__)


class AnimationManager:
    """Manages all camera animations for the network visualization"""

    # Animation names
    COSMIC_ZOOM = "cosmic_zoom"
    ORBIT_FLYBY = "orbit_flyby"
    SPIRAL_DIVE = "spiral_dive"
    BOUNCE_ZOOM = "bounce_zoom"
    SWING_AROUND = "swing_around"
    PULSE_ZOOM = "pulse_zoom"
    MATRIX_EFFECT = "matrix_effect"
    MATRIX_EFFECT_SLOW = "matrix_effect_slow"

    def __init__(self, view):
        """Initialize animation manager with a view"""
        self.view = view
        self._setup_animators()
        self._last_random_anim = None  # Track last random animation to avoid repeats

    def _setup_animators(self):
        """Initialize all animation classes"""
        self.animators = {
            self.COSMIC_ZOOM: CosmicZoomAnimator(self.view),
            self.ORBIT_FLYBY: OrbitFlybyAnimator(self.view),
            self.SPIRAL_DIVE: SpiralDiveAnimator(self.view),
            self.BOUNCE_ZOOM: BounceZoomAnimator(self.view),
            self.SWING_AROUND: SwingAroundAnimator(self.view),
            self.PULSE_ZOOM: PulseZoomAnimator(self.view),
            self.MATRIX_EFFECT: MatrixEffectAnimator(self.view),
            self.MATRIX_EFFECT_SLOW: MatrixEffectSlowAnimator(self.view),
        }

    def get_available_animations(self):
        """Get list of available animation names"""
        return list(self.animators.keys())

    def play_animation(self, animation_name):
        """Play a specific animation by name"""
        if animation_name not in self.animators:
            logger.warning(f"Animation {animation_name} not found")
            return False

        animator = self.animators[animation_name]
        if animator.is_animating():
            logger.debug(f"Animation {animation_name} already in progress")
            return False

        logger.debug(f"Playing animation: {animation_name}")
        animator.animate()
        return True

    def play_random_animation_by_chance(self, chance=1.0):
        """Play a random animation with a given chance (0.0 to 1.0)

        Args:
            chance (float): Probability of playing an animation (0.0 to 1.0)

        Returns:
            bool: True if animation was played, False otherwise
        """
        if random.random() >= chance:
            # bad luck :(
            return False

        # Get all available animations
        animations = self.get_available_animations()

        # If we have a last animation, try to avoid repeating it
        if self._last_random_anim and len(animations) > 1:
            animations.remove(self._last_random_anim)

        # Select and play random animation
        selected_anim = random.choice(animations)
        self._last_random_anim = selected_anim

        logger.debug(f"Playing random animation: {selected_anim}")
        return self.play_animation(selected_anim)

    def is_animating(self):
        """Check if any animation is currently playing"""
        return any(animator.is_animating() for animator in self.animators.values())
