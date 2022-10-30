import torch

print(torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU')
