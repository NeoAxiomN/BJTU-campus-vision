import torch


def _device_summary(
) -> str:
    return (
        f"cuda_available={torch.cuda.is_available()}, "
        f"mps_built={torch.backends.mps.is_built()}, "
        f"mps_available={torch.backends.mps.is_available()}"
    )


def resolve_device(
	requested: str,
) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is not available. {_device_summary()}")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"MPS was requested but is not available. {_device_summary()}")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"Unsupported device option: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
