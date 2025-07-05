import torch
from torch import nn
import torchvision


class ImageClfModel(nn.Module):
    def __init__(
        self, in_channel=1, num_classes=9, backbone="resnet18", pretrained=False
    ):
        super(ImageClfModel, self).__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=pretrained)
        if in_channel != 3:
            # Modify the first conv layer to accept single channel input
            self.backbone.conv1 = nn.Conv2d(
                in_channel,
                self.backbone.conv1.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        logits = self.backbone(x)
        return logits, self.softmax(logits)


if __name__ == "__main__":
    # Example usage
    model = ImageClfModel(in_channel=1, num_classes=9, backbone="resnet18")
    print(model)
    # Test with a random input
    x = torch.randn(1, 1, 224, 224)  # Batch size of 1, single channel input
    logits, probabilities = model(x)
    print(logits.shape, probabilities.shape)
