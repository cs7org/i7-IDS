import torch
from torch import nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = x.mean(-1)  # [B, C]
        y = F.relu(self.fc1(y))  # [B, C//r]
        y = torch.sigmoid(self.fc2(y))  # [B, C]
        return x * y.view(b, c, 1)


class CNN1D(nn.Module):
    def __init__(
        self,
        input_size: int = 46,
        hidden_channels: list[int] = [64, 128, 256, 512, 256, 128, 64],
        output_size: int = 9,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False
                )
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(SqueezeExcite(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        flat = in_ch * input_size
        self.fc = nn.Sequential(
            nn.Linear(flat, flat // 2, bias=False),
            nn.BatchNorm1d(flat // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(flat // 2, output_size),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, features]
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv(x)  # [B, C, L]
        x = x.flatten(1)  # [B, C*L]
        logits = self.fc(x)

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
