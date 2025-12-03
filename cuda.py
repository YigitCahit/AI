import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
