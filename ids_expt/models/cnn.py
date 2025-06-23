import torch
from torch import nn
import torch.nn.functional as F


class CNN1D(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 46,
        hidden_channels: list[int] = [32, 64, 128, 256],
        output_size: int = 9,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = True,
        kernel_size: int = 3,
    ):
        super(CNN1D, self).__init__()
        layers = []
        in_channels = 1  # Treat tabular data as 1-channel sequence
        seq_len = input_size  # Preserve sequence length through padding

        # Conv blocks matching FFNN's hidden layer structure
        for out_channels in hidden_channels:
            layers.append(
                torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(out_channels))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
            in_channels = out_channels

        # Final linear layer matching output size
        self.final_linear = torch.nn.Linear(in_channels * seq_len, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv_layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reshape input: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        x = self.conv_layers(x)

        # Flatten for final linear layer
        x = x.flatten(start_dim=1)
        logits = self.final_linear(x)

        return logits, self.softmax(logits)


class CNN2D(nn.Module):
    def __init__(self, in_channel=1, num_classes=9):
        super(CNN2D, self).__init__()
        # model with global average pooling
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        prob = nn.Softmax(dim=1)
        return x, prob(x)


class BiggerCNN2D(nn.Module):
    def __init__(self, in_channel=1, num_classes=9, dropout_rate=0.3):
        super(BiggerCNN2D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x, F.softmax(x, dim=1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
