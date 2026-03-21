import torch
import torch.nn as nn


class DDSA_CNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_size: tuple = (138, 256),
        encoder_channels: list = [16, 32, 64, 128],
        bottleneck_dim: int = 128,
        sparsity_lambda=1e-5,
        sparsity_target=0.005,
    ):
        """
        Deep Denoising Sparse Autoencoder (DDSA)

        Args:
            input_channels (int): Number of input channels (1 for grayscale)
            input_size (tuple): (H, W) of input images
            encoder_channels (list): List of channel sizes for encoder Conv2D layers
            bottleneck_dim (int): Dimension of the bottleneck dense layer
            sparsity_target (float): Target sparsity level μ (default: 0.1)
        """
        super(DDSA_CNN, self).__init__()

        self.input_channels = input_channels
        self.input_size = input_size
        self.sparsity_target = sparsity_target
        self.sparsity_lambda = sparsity_lambda
        self.bottleneck_dim = bottleneck_dim

        # Encoder: Conv2D layers with SAME padding to preserve spatial dimensions
        encoder_layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(encoder_channels):
            encoder_layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),  # SAME padding
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                ]
            )
            in_channels = out_channels

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate flattened size for dense layers
        self.flat_size = input_size[0] * input_size[1] * encoder_channels[-1]

        # Encoder: Dense layers (bottleneck)
        self.encoder_dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        # Decoder: Dense layers
        self.decoder_dense = nn.Sequential(
            nn.Linear(bottleneck_dim, self.flat_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        # Decoder: Conv2D layers (reverse of encoder) with SAME padding
        decoder_layers = []
        decoder_channels = list(reversed(encoder_channels)) + [input_channels]

        for i in range(len(decoder_channels) - 1):
            decoder_layers.extend(
                [
                    nn.Conv2d(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),  # SAME padding
                    (
                        nn.ReLU(inplace=True)
                        if i < len(decoder_channels) - 2
                        else nn.Sigmoid()
                    ),
                ]
            )
            if i < len(decoder_channels) - 2:
                decoder_layers.append(nn.BatchNorm2d(decoder_channels[i + 1]))

        self.decoder_conv = nn.Sequential(*decoder_layers)

        # For sparsity constraint tracking
        self.register_buffer("running_mean_activation", torch.zeros(bottleneck_dim))

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        encoded_conv = self.encoder_conv(
            x
        )  # Shape: (batch, encoder_channels[-1], H, W)
        encoded_flat = self.encoder_dense(
            encoded_conv
        )  # Shape: (batch, bottleneck_dim)

        # Track activations for sparsity constraint (only during training)
        if self.training:
            mean_activation = torch.mean(encoded_flat, dim=0)
            self.running_mean_activation = (
                0.999 * self.running_mean_activation + 0.001 * mean_activation.detach()
            )

        # Decoder
        decoded_flat = self.decoder_dense(encoded_flat)  # Shape: (batch, flat_size)
        # Reshape back to feature map dimensions
        decoded_reshaped = decoded_flat.view(
            batch_size,
            self.encoder_conv[-3].out_channels,
            self.input_size[0],
            self.input_size[1],
        )
        output = self.decoder_conv(
            decoded_reshaped
        )  # Shape: (batch, input_channels, H, W)

        return output, encoded_flat

    def sparsity_penalty(self, encoded):
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        epsilon = 1e-4
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
            (1 - rho) / (1 - rho_hat)
        )
        sparsity_penalty = torch.sum(kl_divergence)
        return sparsity_penalty * self.sparsity_lambda


# Example usage for your 138×256 grayscale images
if __name__ == "__main__":
    # Configuration
    input_channels = 1
    input_size = (138, 256)
    encoder_channels = [32, 64, 128]  # Customize as needed
    bottleneck_dim = 256

    # Create model
    model = DDSA_CNN(
        input_channels=input_channels,
        input_size=input_size,
        encoder_channels=encoder_channels,
        bottleneck_dim=bottleneck_dim,
    )

    print("DDSA Model Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)

    # Test with dummy data
    dummy_input = torch.randn(2, 1, 138, 256)
    with torch.no_grad():
        output, hidden = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden representation shape: {hidden.shape}")

    # Verify exact shape match
    assert (
        output.shape == dummy_input.shape
    ), f"Shape mismatch! Expected {dummy_input.shape}, got {output.shape}"
    print("✓ Output shape exactly matches input shape!")

    # Test sparsity loss
    sparsity_loss_value = model.sparsity_loss(hidden)
    print(f"Sparsity loss: {sparsity_loss_value.item():.6f}")
