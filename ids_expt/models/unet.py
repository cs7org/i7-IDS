import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetAE(smp.Unet):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = None,
        in_channels: int = 1,
        out_channels: int = 1,
        activation: str = "sigmoid",
        drop_encoder_decoder_connnection: bool = False,
        drop_first_n_encoder_connections: int = 1,
    ):
        """
        U-Net based purifier for adversarial images.

        Args:
            encoder_name: name of the backbone (e.g., "resnet34", "efficientnet-b0").
            encoder_weights: pretrained weights for the backbone ("imagenet" or None).
            in_channels: number of channels in the input images.
            out_channels: number of channels in the output (should match in_channels).
            activation: final activation ("sigmoid" for [0,1] outputs).

            prunning top skip connection yields best result than all
            https://arxiv.org/pdf/2402.08276

        """
        self.drop_encoder_decoder_connnection = drop_encoder_decoder_connnection
        self.drop_first_n_encoder_connections = drop_first_n_encoder_connections
        super().__init__(
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
        encoder_outputs = self.encoder(x)
        self.encoder_outputs = encoder_outputs
        if self.drop_encoder_decoder_connnection:
            # zero out the first n encoder outputs
            for i in range(
                min(self.drop_first_n_encoder_connections, len(encoder_outputs))
            ):
                if i < len(encoder_outputs):
                    encoder_outputs[i] = torch.zeros_like(encoder_outputs[i])

        x = self.decoder(encoder_outputs)

        self.decoder_outputs = x

        return self.segmentation_head(x)


if __name__ == "__main__":
    purifier = UnetAE(
        encoder_name="resnet18",
        in_channels=1,
        out_channels=1,
        activation="sigmoid",
        drop_encoder_decoder_connnection=True,
    )
    dummy = torch.randn(4, 1, 138, 256)
    purified = purifier(dummy)
    print("Input shape:   ", dummy.shape)
    print("Output shape:  ", purified.shape)

    print("Encoder outputs:", [o.shape for o in purifier.encoder_outputs])
    print("Decoder outputs:", purifier.decoder_outputs.shape)
