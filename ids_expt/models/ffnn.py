import torch

class DNP3DNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNP3DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, 90),
            nn.ReLU(),
            nn.Linear(90, output_dim),
            # nn.Softmax(dim=1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return x, self.softmax(x)



class FFNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 46,
        hidden_layers: list[int] = [90] * 10,
        output_size: int = 9,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = True,
    ):
        super(FFNN, self).__init__()
        layers = []
        current_size = input_size

        for hidden_size in hidden_layers:
            layers.append(torch.nn.Linear(current_size, hidden_size))
            if use_batchnorm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
            current_size = hidden_size

        layers.append(torch.nn.Linear(current_size, output_size))

        self.softmax = torch.nn.Softmax(dim=1)
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x, self.softmax(x)
