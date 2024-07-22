from .inference import SeedStoryInferenceNode
from .model_loader import ModelLoaderNode

NODE_CLASS_MAPPINGS = {
    "SeedStoryInferenceNode": SeedStoryInferenceNode,
    "ModelLoaderNode": ModelLoaderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedStoryInferenceNode": "Seed Story Inference Node",
    "ModelLoaderNode": "Model Loader Node"
}

