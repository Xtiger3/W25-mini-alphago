import torch.nn as nn

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

