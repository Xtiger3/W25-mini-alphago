import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, kernel, stride)
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel, stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        # 2. Batch normalisation
        # 3. A rectifier non-linearity
        out = self.conv(x)
        # 4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        out = self.conv2d(out)
        # 5. Batch normalisation
        out = self.norm(out)
        # 6. A skip connection that adds the input to the block
        out = out + x
        # 7. A rectifier non-linearity
        out = self.relu(out)

        return out

    
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()

        assert(kernel % 2 == 1) 

        padding = (kernel-1) // 2
    
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        return out
    


if __name__ == "__main__":
    conv_nn = ConvBlock(9, 1, 3, 1) 

    #TODO: Add code for creating child node of the board, and pass the tensor generated from it through the network

