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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()

        padding = (kernel-1) // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        """ based on: https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf#page=27 """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + x
        out = self.relu(out)

        return out


if __name__ == "__main__":
    conv_nn = ConvBlock(9, 1, 3, 1) 

    #TODO: Add code for creating child node of the board, and pass the tensor generated from it through the network

