import importlib

__attributes = {
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureSceneContextFlowModel": "sparse_structure_sc_flow",
    "SLatGaussianDecoder": "structured_latent_vae.decoder_gs",
    "SLatMeshDecoder": "structured_latent_vae.decoder_mesh",
    "SLatFlowModel": "structured_latent_flow",
    "ImageConditioner": "image_conditioner",
}

__all__ = list(__attributes.keys())

def __getattr__(name):
    if name not in __attributes:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_name = __attributes[name]
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def from_pretrained(path: str, revision: str | None = None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json", revision=revision)
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors", revision=revision)

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file))

    return model
