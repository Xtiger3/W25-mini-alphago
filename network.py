import torch.nn as nn
import math

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
        # 1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        out = self.conv1(x)
        # 2. Batch normalisation
        out = self.norm1(out)
        # 3. A rectifier non-linearity
        out = self.relu(out)
        # # 4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        out = self.conv2(out)
        # # 5. Batch normalisation
        out = self.norm2(out)
        # # 6. A skip connection that adds the input to the block
        out = out + x
        # # 7. A rectifier non-linearity        
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


class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, 256, 3, 1)
        self.res_blocks = nn.Sequential(*[ResBlock(256, 256, 3, 1) for _ in range(19)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions),
            # nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * 9, 256),
            nn.ReLU(),
            # A fully connected linear layer to a scalar
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.init_weights()  # Initialize weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1 / math.sqrt(m.kernel_size[0] ** 2 * m.in_channels))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1 / math.sqrt(m.in_features))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


if __name__ == "__main__":
    conv_nn = ConvBlock(9, 1, 3, 1) 

    #TODO: Add code for creating child node of the board, and pass the tensor generated from it through the network

