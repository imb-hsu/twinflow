import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


class AutoencoderModel(nn.Module):
    """Simple autoencoder for anomaly detection."""

    def __init__(self, input_dim, encoding_dim=32):
        super(AutoencoderModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDetector:
    """Anomaly detector using autoencoder reconstruction error."""

    def __init__(self, input_dim, encoding_dim=32, learning_rate=0.001, device="cpu"):
        self.device = device
        self.model = AutoencoderModel(input_dim, encoding_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        self.threshold = None

    def train(self, X_train, epochs=50, batch_size=32):
        """Train the autoencoder on normal data."""
        # Normalize data
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create data loader
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_losses = []
            for batch in train_loader:
                data = batch[0].to(self.device)

                # Forward pass
                reconstructed = self.model(data)
                loss = self.criterion(reconstructed, data)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        # Calculate threshold based on training reconstruction error
        self.model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            reconstructed = self.model(X_train_tensor)
            reconstruction_errors = torch.mean(
                (X_train_tensor - reconstructed) ** 2, dim=1
            )
            self.threshold = torch.mean(reconstruction_errors) + 3 * torch.std(
                reconstruction_errors
            )
            self.threshold = self.threshold.item()

        print(f"Training completed. Threshold set to: {self.threshold:.6f}")
        return losses

    def predict(self, X_test):
        """Predict anomalies based on reconstruction error."""
        X_test_scaled = self.scaler.transform(X_test)

        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            reconstructed = self.model(X_test_tensor)
            reconstruction_errors = torch.mean(
                (X_test_tensor - reconstructed) ** 2, dim=1
            )
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        # Anomaly: reconstruction error > threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)

        return predictions, reconstruction_errors

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        predictions, reconstruction_errors = self.predict(X_test)

        # Calculate metrics
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        # Calculate AUC if possible
        try:
            auc = roc_auc_score(y_test, reconstruction_errors)
        except:
            auc = None

        results = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "threshold": self.threshold,
        }

        return results, predictions, reconstruction_errors


def load_data(data_path):
    """Load data from CSV files."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    # Load all CSV files
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_path}")

    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    # Concatenate all dataframes
    data = pd.concat(dataframes, ignore_index=True)

    return data


def prepare_data(data, label_column="label"):
    """Prepare data for training/testing."""
    # Separate features and labels
    if label_column in data.columns:
        X = data.drop(columns=[label_column])
        y = data[label_column].values
    else:
        X = data
        y = None

    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Handle missing values
    X = X.fillna(X.mean())

    return X.values, y


def main():
    """Main function to run anomaly detection."""
    print("Starting Autoencoder-based Anomaly Detection")
    print("=" * 60)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training data (0pct_faults)
    print("\nLoading training data from data/0pct_faults...")
    train_data = load_data("data/0pct_faults")
    X_train, _ = prepare_data(train_data)
    print(f"Training data shape: {X_train.shape}")

    # Load test data
    print("\nLoading test data from data/test...")
    test_data = load_data("data/test")
    X_test, y_test = prepare_data(test_data)
    print(f"Test data shape: {X_test.shape}")

    if y_test is None:
        print("Warning: No labels found in test data. Creating dummy labels.")
        y_test = np.zeros(len(X_test))

    # Initialize and train detector
    print("\nInitializing anomaly detector...")
    input_dim = X_train.shape[1]
    detector = AnomalyDetector(
        input_dim=input_dim, encoding_dim=32, learning_rate=0.001, device=device
    )

    print("\nTraining autoencoder...")
    losses = detector.train(X_train, epochs=50, batch_size=32)

    # Evaluate on test data
    print("\nEvaluating on test data...")
    results, predictions, reconstruction_errors = detector.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    if results["auc"] is not None:
        print(f"AUC: {results['auc']:.4f}")
    print(f"Threshold: {results['threshold']:.6f}")
    print(f"Detected anomalies: {np.sum(predictions)}/{len(predictions)}")
    print("=" * 60)

    # Save model
    print("\nSaving model...")
    torch.save(
        {
            "model_state_dict": detector.model.state_dict(),
            "scaler": detector.scaler,
            "threshold": detector.threshold,
            "input_dim": input_dim,
        },
        "autoencoder_model.pth",
    )
    print("Model saved to autoencoder_model.pth")


if __name__ == "__main__":
    main()
