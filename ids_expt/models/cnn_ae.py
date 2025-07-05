import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DecoderNoSkip(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        # The first decoder block should take the encoder's bottleneck output as input
        in_channels = encoder_channels[-1]
        self.blocks = nn.ModuleList()
        for out_channels in decoder_channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class UNetAutoencoderNoSkip(nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        in_channels=3,
        out_channels=3,
        encoder_weights=None,
    ):
        super().__init__()
        # Get encoder from SMP
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels

        decoder_channels = encoder_channels[-2::-1]
        self.decoder = DecoderNoSkip(encoder_channels, decoder_channels)
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        x = self.decoder(x)
        x = self.final_conv(x)
        return x


# Example usage:
if __name__ == "__main__":
    model = UNetAutoencoderNoSkip(
        encoder_name="resnet18",
        in_channels=1,
        out_channels=1,
        encoder_weights=None,
    )
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
