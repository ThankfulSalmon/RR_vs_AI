import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random

def simulate_slam_pose(index):
    radius = 10
    angle = index * np.pi / 30
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 1.5  # fixed height
    return np.array([x, y, z])

def apply_fog(image, intensity=0.5):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    fog_layer = Image.new("RGB", image.size, (255, 255, 255))
    blended = Image.blend(image, fog_layer, alpha=intensity)
    fogged_image = blended.filter(ImageFilter.GaussianBlur(radius=3))
    return fogged_image

def apply_low_visibility(image, intensity=0.3):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(1 - intensity)

    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(1 - intensity)
    return image

def apply_low_friction(current_velocity, friction_coefficient=0.3, time_step=0.1):
    friction_force = -friction_coefficient * current_velocity
    new_velocity = current_velocity + friction_force * time_step
    return new_velocity

def apply_random_weather_effects(image):
    effects = [lambda img: img, apply_fog, apply_low_visibility]
    effect = random.choice(effects)
    return effect(image)
