import torch

class CudaDevice():
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        self.device = torch.cuda.device(0)
        self.get_device_name = torch.cuda.get_device_name(0)
        
if __name__ == '__main__':
    cuda_device = CudaDevice()
    cuda_device.device_count