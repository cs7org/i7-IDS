import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from loguru import logger
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int = 72, projection: int = 72):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, projection),
        )
        self.decoder = nn.Sequential(
            nn.Linear(projection, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
        )

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
