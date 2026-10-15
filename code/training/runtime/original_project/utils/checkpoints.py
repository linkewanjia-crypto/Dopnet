from pathlib import Path
import torch

def load_checkpoint(path, map_location="cpu"):
    """Load a trusted project checkpoint containing model weights + Python config metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)

def save_checkpoint(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)
