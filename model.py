import os
import pandas as pd
from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from monai.utils import set_determinism
from sklearn.utils.class_weight import compute_class_weight
from ignite.metrics import Accuracy, Precision, Recall, Fbeta
import numpy as np
import wandb
from ignite.engine import Events
from tqdm import tqdm
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from torch.utils.data import WeightedRandomSampler


# Helper classes for logging and checkpointing
class WandbMetricsLogger:
    def __init__(self, prefix="train"):
        self.prefix = prefix

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics)

    def log_metrics(self, engine):
        metrics = engine.state.metrics
        epoch = engine.state.epoch
        log_dict = {f"{self.prefix}/{k}": v for k, v in metrics.items()}
        log_dict[f"{self.prefix}/epoch"] = epoch
        wandb.log(log_dict)


class WandbLRLogger:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.log_lr)

    def log_lr(self, engine):
        epoch = engine.state.epoch
        lr = self.optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": lr, "epoch": epoch})


class WandbModelCheckpointHandler:
    def __init__(self, save_dir, model, metric_name="val_rocauc"):
        self.save_dir = save_dir
        self.model = model
        self.metric_name = metric_name

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.save_checkpoint)

    def save_checkpoint(self, engine):
        epoch = engine.state.epoch
        metric = engine.state.metrics.get(self.metric_name)
        checkpoint_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(
            artifact, aliases=[f"epoch_{epoch}", f"{self.metric_name}_{metric:.4f}"]
        )


# Custom dataset for BIRADS classification
class BIRADSDataset(Dataset):
    def __init__(self, dataframe, transform=None, image_dir=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]["Image_filename"]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)

        if self.transform:
            image = self.transform(image)

        birads_label = self.dataframe.iloc[idx]["BIRADS_encoded"]
        return image, torch.tensor(birads_label, dtype=torch.long)


# Classifier model
class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout(config.classifier_dropout)
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        return self.linear(x)


# Backbone model using ResNet50
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        return self.backbone(x)


# Setup device, seed, and W&B
device = torch.device("cuda")
model_seed = 42
torch.manual_seed(model_seed)
np.random.seed(model_seed)
set_determinism(seed=model_seed)

# Initialize W&B
wandb.init(
    project="BIRADS-Classification",
    config={
        "model": "EfficientNetV2-S",
        "pretrained": False,
        "num_classes": 6,
        "input_size": (384, 384),
        "batch_size": 16,
        "epochs": 30,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.1,
        "scheduler_patience": 3,
        "classifier_dropout": 0.3,
        "optimizer": "Adam",
        "loss_function": "WeightedCrossEntropy",
        "scheduler": "ReduceLROnPlateau",
        "seed": model_seed,
        "augmentation": ["RandFlip", "RandRotate", "RandZoom", "RandIntensityShift"],
    },
)
config = wandb.config

# Load and prepare data
df = pd.read_excel("birads_dataset_filtered_cropped_images.xlsx")
birads_mapping = {
    2: 0, 
    3: 1, 
    "4a": 2, 
    "4b": 3, 
    "4c": 4, 
    5: 5
    }
df["BIRADS_encoded"] = df["BIRADS"].map(birads_mapping)

# Train-validation split
train_df, val_df = train_test_split(
    df, test_size=0.15, stratify=df["BIRADS_encoded"], random_state=model_seed
)

# Class and sample weights
labels = train_df["BIRADS_encoded"].values
class_weights = torch.tensor(
    compute_class_weight("balanced", classes=np.unique(labels), y=labels),
    dtype=torch.float,
).to(device)
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

# Define data transforms
radimagenet_mean, radimagenet_std = [0.488] * 3, [0.246] * 3
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        #transforms.Normalize(mean=radimagenet_mean, std=radimagenet_std),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=radimagenet_mean, std=radimagenet_std),
    ]
)

# Create datasets and loaders
train_dataset = BIRADSDataset(
    train_df, transform=train_transforms, image_dir="./cropped_cases_grayscale/"
)
val_dataset = BIRADSDataset(
    val_df, transform=val_transforms, image_dir="./cropped_cases_grayscale/"
)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

# Initialize model components
backbone = Backbone()
classifier = Classifier(num_class=6)
backbone.load_state_dict(torch.load("./pretrained_models/ResNet50.pt"))
model = nn.Sequential(backbone, classifier).to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Define optimizer, loss, scheduler, and metrics
optimizer = optim.Adam(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config.scheduler_factor,
    patience=config.scheduler_patience,
)
criterion = nn.CrossEntropyLoss()

metrics = {
    "accuracy": Accuracy(),
    "precision": Precision(average=False),
    "recall": Recall(average=False),
    "fbeta": Fbeta(beta=1.0, average=False),
}

# Start training
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for metric in metrics.values():
        metric.reset()

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        for metric in metrics.values():
            metric.update((outputs, labels))

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_metrics = {name: metric.compute() for name, metric in metrics.items()}

    mean_precision = torch.mean(epoch_metrics['precision']).item()
    mean_recall = torch.mean(epoch_metrics['recall']).item()
    mean_f1 = torch.mean(epoch_metrics['fbeta']).item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_metrics['accuracy']:.4f}, "
          f"Mean Precision: {mean_precision:.4f}, Mean Recall: {mean_recall:.4f}, Mean F1: {mean_f1:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for metric in metrics.values():
            metric.reset()

        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            for metric in metrics.values():
                metric.update((outputs, labels))

    val_loss /= len(val_loader.dataset)
    val_metrics = {name: metric.compute() for name, metric in metrics.items()}
    scheduler.step(val_loss)

    mean_val_precision = torch.mean(val_metrics['precision']).item()
    mean_val_recall = torch.mean(val_metrics['recall']).item()
    mean_val_f1 = torch.mean(val_metrics['fbeta']).item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, "
          f"Accuracy: {val_metrics['accuracy']:.4f}, "
          f"Mean Precision: {mean_val_precision:.4f}, Mean Recall: {mean_val_recall:.4f}, Mean F1: {mean_val_f1:.4f}")

    # Log metrics with W&B
    wandb.log({
        'train_loss': epoch_loss,
        'train_accuracy': epoch_metrics['accuracy'],
        'mean_train_precision': mean_precision,
        'mean_train_recall': mean_recall,
        'mean_train_fbeta': mean_f1,
        'val_loss': val_loss,
        'val_accuracy': val_metrics['accuracy'],
        'mean_val_precision': mean_val_precision,
        'mean_val_recall': mean_val_recall,
        'mean_val_fbeta': mean_val_f1,
    })

wandb.finish()
