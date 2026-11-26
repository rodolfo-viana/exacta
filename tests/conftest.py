import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request) -> torch.device:
    return torch.device(request.param)
