import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from loguru import logger
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F


class DDSA_FFNN(nn.Module):

    def __init__(
        self,
        input_size: int = 37,
        hidden_sizes: list = [16, 32, 64, 128],
        bottleneck_dim: int = 128,
        sparsity_lambda: float = 1e-4,
        sparsity_target: float = 0.05,
        dropout_rate: float = 0.25,
    ):
        """
        Deep Denoising Sparse Autoencoder (DDSA) - Feedforward Neural Network

        Args:
            input_size (int): Dimension of flattened input (e.g., 784 for 28x28 images)
            hidden_sizes (list): List of hidden layer sizes for encoder
            bottleneck_dim (int): Dimension of the bottleneck dense layer
            sparsity_lambda (float): Weight for sparsity penalty
            sparsity_target (float): Target sparsity level ρ (default: 0.05)
            dropout_rate (float): Dropout rate for regularization
        """
        super(DDSA_FFNN, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.bottleneck_dim = bottleneck_dim
        self.sparsity_target = sparsity_target
        self.sparsity_lambda = sparsity_lambda
        self.dropout_rate = dropout_rate

        # Encoder: Dense layers with ReLU activation
        encoder_layers = []
        in_features = input_size

        for i, out_features in enumerate(hidden_sizes):
            encoder_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(out_features),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = out_features

        # Bottleneck layer
        encoder_layers.extend(
            [
                nn.Linear(in_features, bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
        )

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: Dense layers (reverse of encoder)
        decoder_layers = []
        in_features = bottleneck_dim

        for i, out_features in enumerate(reversed(hidden_sizes)):
            decoder_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(out_features),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = out_features

        # Output layer
        decoder_layers.extend(
            [
                nn.Linear(in_features, input_size),
                nn.Sigmoid(),  # For normalized inputs [0,1]
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # For sparsity constraint tracking
        self.register_buffer("running_mean_activation", torch.zeros(bottleneck_dim))

    def forward(self, x):
        batch_size = x.size(0)

        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Encoder
        encoded = self.encoder(x)  # Shape: (batch, bottleneck_dim)

        # Track activations for sparsity constraint (only during training)
        if self.training:
            mean_activation = torch.mean(encoded, dim=0)
            self.running_mean_activation = (
                0.999 * self.running_mean_activation + 0.001 * mean_activation.detach()
            )

        # Decoder
        decoded = self.decoder(encoded)  # Shape: (batch, input_size)

        return decoded, encoded

    def sparsity_penalty(self, encoded):
        """
        Calculate KL divergence sparsity penalty as described in the DDSA paper
        """
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        epsilon = 1e-4
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log(
            (1 - rho) / (1 - rho_hat)
        )
        sparsity_penalty = torch.sum(kl_divergence)
        return sparsity_penalty * self.sparsity_lambda

    def project(self, x):
        """
        Get the encoded representation (bottleneck features)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 72,
        hidden_sizes: list = [64, 128, 256],
        projection: int = 72,
    ):
        super(AutoEncoder, self).__init__()

        encoder_layers = []
        input_dim = input_size
        for hs in hidden_sizes:
            encoder_layers.append(nn.Linear(input_dim, hs))
            encoder_layers.append(nn.ReLU())
            input_dim = hs
        encoder_layers.append(nn.Linear(input_dim, projection))
        encoder_layers.append(nn.ReLU())

        decoder_layers = []
        input_dim = projection
        for hs in reversed(hidden_sizes):
            decoder_layers.append(nn.Linear(input_dim, hs))
            decoder_layers.append(nn.ReLU())
            input_dim = hs
        decoder_layers.append(nn.Linear(input_dim, input_size))
        decoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def project(self, x):
        return self.encoder(x)


class AutoEncoderTrainer:
    def __init__(
        self,
        input_size: int,
        projection_dim: int,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
        min_max_scale: bool = True,
        log_every: int = 1,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.log_every = log_every
        self.min_max_scale = min_max_scale
        self.projection_dim = projection_dim
        self.input_size = input_size
        self.model = None

    def before_train(self):
        logger.info("Before Training")
        self.model = AutoEncoder(
            input_size=self.input_size, projection=self.projection_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.metrics = {
            "train_loss": [],
            "test_loss": [],
            "train_time": [],
            "epoch": [],
        }

    def train(self, train_features: pd.DataFrame, test_features: pd.DataFrame):
        self.before_train()
        logger.info("Starting Training")
        X_train = torch.tensor(train_features.values, dtype=torch.float32)
        X_test = torch.tensor(test_features.values, dtype=torch.float32)

        if self.min_max_scale:
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)

        X_train = X_train.to(self.device)
        X_test = X_test.to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train), batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test), batch_size=self.batch_size, shuffle=False
        )

        logger.info(f"Training on {self.device}")
        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            start_time = time.time()

            with tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]"
            ) as pbar:
                for (batch,) in pbar:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs, batch)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

            avg_train_loss = epoch_loss / len(train_loader)
            end_time = time.time()

            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                with tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Test]"
                ) as pbar:
                    for (batch,) in pbar:
                        batch = batch.to(self.device)
                        outputs = self.model(batch)
                        loss = self.loss_fn(outputs, batch)
                        test_loss += loss.item()
                        pbar.set_postfix(loss=loss.item())

            avg_test_loss = test_loss / len(test_loader)

            self.metrics["train_loss"].append(avg_train_loss)
            self.metrics["test_loss"].append(avg_test_loss)
            self.metrics["train_time"].append(end_time - start_time)
            self.metrics["epoch"].append(epoch)

            if epoch % self.log_every == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}"
                )

        return self.metrics

    def project(self, data: pd.DataFrame):
        if self.min_max_scale:
            data = self.scaler.transform(data)
            data = torch.tensor(data, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            X = data.to(self.device)
            encoded = self.model.project(X).cpu().numpy()
        return encoded


# for df in [dnp3_df, numerical_df]:
#     features_df = df.drop(columns=["Label"])
#     labels_df = df["Label"]
#     train_features_df, test_features, train_labels_df, test_labels = train_test_split(
#         features_df,
#         labels_df,
#         test_size=0.2,
#         random_state=42,
#         stratify=labels_df,
#     )
#     trainer = AutoEncoderTrainer(
#         input_size=features_df.shape[1],
#         projection_dim=features_df.shape[1] // 2,
#         epochs=5,
#         batch_size=32,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         min_max_scale=True,
#         log_every=10,
#     )

#     trainer.train(train_features=features_df, test_features=test_features)


# Example usage
if __name__ == "__main__":

    # Create model
    model = DDSA_FFNN(
        sparsity_lambda=1e-4,
        sparsity_target=0.05,
        dropout_rate=0.25,
    )

    print("DDSA FFNN Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)

    # Test with dummy data
    batch_size = 32
    dummy_input = torch.randn(batch_size, 37)

    # Forward pass
    with torch.no_grad():
        reconstructed, encoded = model(dummy_input)
        projected = model.project(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Projected shape: {projected.shape}")

    # Verify shapes
    assert reconstructed.shape == dummy_input.shape
    assert encoded.shape == projected.shape
    print("✓ All shapes verified correctly!")
