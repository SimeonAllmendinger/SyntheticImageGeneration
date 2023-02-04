import torch

class CudaDevice():
    
    @property
    def available(self):
        return torch.cuda.is_available()

    @property
    def device_count(self):
        return torch.cuda.device_count()
    
    @property
    def current_device(self):
        return torch.cuda.current_device()
    
    @property
    def device_name_0(self):
        return torch.cuda.get_device_name(0)
    
        
if __name__ == '__main__':
    cuda_device = CudaDevice()
    print('Device Name: ' + cuda_device.device_name)