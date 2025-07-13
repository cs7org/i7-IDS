import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetAE(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 1,
        out_channels: int = 1,
        activation: str = "sigmoid",
    ):
        """
        U-Net based purifier for adversarial images.

        Args:
            encoder_name: name of the backbone (e.g., "resnet34", "efficientnet-b0").
            encoder_weights: pretrained weights for the backbone ("imagenet" or None).
            in_channels: number of channels in the input images.
            out_channels: number of channels in the output (should match in_channels).
            activation: final activation ("sigmoid" for [0,1] outputs).
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch, in_channels, H, W)

        Returns:
            purified: tensor of shape (batch, out_channels, H, W)
        """
        return self.model(x)


if __name__ == "__main__":
    purifier = UnetAE(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        out_channels=1,
        activation="sigmoid",
    )
    dummy = torch.randn(4, 1, 138, 256)
    purified = purifier(dummy)
    print("Input shape:   ", dummy.shape)
    print("Output shape:  ", purified.shape)
