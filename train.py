import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ----------------------------
# CONFIG
# ----------------------------
CSV_FILE = "plant_disease_weather_data.csv"
MODEL_SAVE_PATH = "ecovision_model.pth"
SCALER_SAVE_PATH = "scaler.save"
LABEL_ENCODER_PATH = "label_encoder.save"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.getcwd()

# ----------------------------
# LOAD DATA
# ----------------------------
data = pd.read_csv(CSV_FILE)

print("Original dataset size:", len(data))

env_cols = [
    "avg_temp",
    "humidity",
    "rainfall",
    "sunlight_hours",
    "wind_speed",
    "leaf_wetness"
]

# ----------------------------
# FIX IMAGE PATHS
# ----------------------------
def fix_path(path):
    path = path.replace("\\", "/")
    full_path = os.path.join(BASE_DIR, path)
    return os.path.normpath(full_path)

data["fixed_path"] = data["image_path"].apply(fix_path)

# Remove missing images
data = data[data["fixed_path"].apply(os.path.exists)]

print("Dataset size after removing missing images:", len(data))

if len(data) == 0:
    print("❌ No valid images found. Check your dataset folder!")
    exit()

# ----------------------------
# SCALE ENV FEATURES
# ----------------------------
scaler = StandardScaler()
data[env_cols] = scaler.fit_transform(data[env_cols])
joblib.dump(scaler, SCALER_SAVE_PATH)

# ----------------------------
# ENCODE LABELS
# ----------------------------
label_encoder = LabelEncoder()
data["season_label"] = label_encoder.fit_transform(data["season"])
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

num_classes = len(label_encoder.classes_)
print("Seasons:", label_encoder.classes_)

# ----------------------------
# DATASET CLASS
# ----------------------------
class EcoVisionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["fixed_path"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        env_features = torch.tensor(row[env_cols].values, dtype=torch.float32)
        label = torch.tensor(row["season_label"], dtype=torch.long)

        return image, env_features, label


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = EcoVisionDataset(data, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# MODEL
# ----------------------------
class EcoVisionModel(nn.Module):
    def __init__(self, num_classes, env_feature_size):
        super(EcoVisionModel, self).__init__()

        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.fc1 = nn.Linear(512 + env_feature_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, image, env):
        img_features = self.cnn(image)
        combined = torch.cat((img_features, env), dim=1)
        x = self.relu(self.fc1(combined))
        output = self.fc2(x)
        return output


model = EcoVisionModel(num_classes, len(env_cols)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# TRAINING LOOP
# ----------------------------
print("Training started...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, env_features, labels in dataloader:
        images = images.to(DEVICE)
        env_features = env_features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, env_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(dataloader):.4f}")

print("Training complete!")

# ----------------------------
# SAVE MODEL
# ----------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ Model saved as", MODEL_SAVE_PATH)
