
import json
import pathlib
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model as load_model_torch

from kneel.model import HourglassNet

def load_models(hf_token, revision, save_to="kneel_cache", stage=None):
    """Returns the models from the Hugging Face Hub. Models are loaded in the `eval` mode.
    
    Parameters
    ----------
    revision : str
        The revision of the model to download
    save_to : str
        The path to save the models (cache)
    stage : str, optional
        The stage of the model to download. Default is None, which downloads both stages of the pipeline.
    """
    if stage is None:
        stage = ["global_search", "local_search"]
    else:
        stage = [stage, ]
        
    save_to = pathlib.Path(save_to)
    # Mean and std
    hf_hub_download("imeds/kneel", revision=revision, 
                    filename="mean_std.npy",  
                    cache_dir=save_to, token=hf_token)
    
    mean_std = np.load(save_to / f"models--imeds--kneel/snapshots/{revision}/mean_std.npy")
    models = {}
    for stage in stage:
        models[stage] = []
        for fold in range(5):
            # Config (arguments of the class)
            hf_hub_download("imeds/kneel", revision=revision, filename="config.json", 
                            subfolder=f"{stage}/fold_{fold}", cache_dir=save_to, token=hf_token)
            # Model
            hf_hub_download("imeds/kneel", revision=revision, filename="model.safetensors", 
                            subfolder=f"{stage}/fold_{fold}", cache_dir=save_to, token=hf_token)
            # Intializing the model
            basedir = save_to / f"models--imeds--kneel/snapshots/{revision}/{stage}/fold_{fold}"
            with open(basedir / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            net = HourglassNet(**config)
            load_model_torch(net, basedir / "model.safetensors")
            models[stage].append(net)
    return models, mean_std