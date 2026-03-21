# https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
import torch
from torch import nn
import torch.nn.functional as F


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = in_planes * stride
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = int(in_planes / stride)
        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, input_shape, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=1):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

        # Calculate the size after all conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, nc, *input_shape)
            x = torch.relu(self.bn1(self.conv1(dummy)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            self.final_feature_size = x.view(1, -1).shape[1]

        self.linear = nn.Linear(self.final_feature_size, 2 * z_dim)

    def _make_layer(self, block, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, : self.z_dim]
        logvar = x[:, self.z_dim :]
        return mu, logvar


class ResNet18Dec(nn.Module):
    def __init__(self, input_shape, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=1):
        super().__init__()
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.nc = nc

        # Start with the highest number of channels
        self.in_planes = 512

        # Linear layer to expand latent code
        self.linear = nn.Linear(z_dim, 512)

        # Decoder layers - FIXED: Don't reverse the strides
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)

        # Final upsampling and output layer
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, block, planes, num_Blocks, stride):
        # FIXED: Use forward order, not reversed
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)

        # Start with a small feature map and gradually upsample
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))

        # Ensure output matches input size
        if x.shape[2:] != self.input_shape:
            x = F.interpolate(
                x, size=self.input_shape, mode="bilinear", align_corners=False
            )

        return x


class VAE(nn.Module):
    def __init__(self, input_shape=(138, 256), z_dim=512, nc=1):
        super().__init__()
        self.input_shape = input_shape
        self.encoder = ResNet18Enc(input_shape=input_shape, z_dim=z_dim, nc=nc)
        self.decoder = ResNet18Dec(input_shape=input_shape, z_dim=z_dim, nc=nc)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, z, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def loss_function(self, x, x_recon, mean, logvar, beta=0.8):
        """
        VAE loss function combining reconstruction loss and KL divergence
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss


# Example usage
if __name__ == "__main__":
    # Test with your input shape
    input_shape = (138, 256)
    z_dim = 128
    nc = 1  # Grayscale

    model = VAE(input_shape=input_shape, z_dim=z_dim, nc=nc)

    print("VAE Model Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)

    # Test with dummy input
    batch_size = 4
    x = torch.randn(batch_size, nc, *input_shape)

    # Forward pass
    with torch.no_grad():
        x_recon, z, mean, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent z shape: {z.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Test loss computation
    total_loss, recon_loss, kl_loss = model.loss_function(x, x_recon, mean, logvar)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence loss: {kl_loss.item():.4f}")

    # Verify shapes
    assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} != {x.shape}"
    print("✓ All shapes verified successfully!")
