import torch

DTYPE = torch.float32


def get_device():
    device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")
    return device
