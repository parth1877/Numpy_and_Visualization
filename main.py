import os
import torch
import cv2
import albumentations
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import functional as F
import pretrainedmodels
import torch.nn as nn
from timeit import default_timer as timer
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Adjust the value

save_interval = 100

class ClassificationDataLoader(Dataset):
    def __init__(self, image_paths, targets, resize=None, augmentations=None, mean=None, std=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.resize:
                image = cv2.resize(
                    image, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA
                )

            if self.augmentations:
                augmented = self.augmentations(image=image)
                image = augmented["image"]

            # Convert image to PyTorch tensor and normalize
            image = transforms.ToTensor()(image)
            if self.mean is not None and self.std is not None:
                image = transforms.functional.normalize(image, mean=self.mean, std=self.std)

            target = self.targets[idx]
            return {
                "image": image,
                "target": torch.tensor(target, dtype=torch.float),
            }
        except Exception as e:
            print(f"Error loading image: {image_path}. Error: {str(e)}")
            return None

class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1, 1).type_as(out)
        )
        return out, loss

model = None
model_path = "C:\\skin_cancer"


def train(fold):
    start = timer()
    training_data_path = "C:\\skin_cancer\\train224"
    df = pd.read_csv("C:\\skin_cancer\\train_folds.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    train_bs = 5
    valid_bs = 5
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    print(f"Training on fold {fold}")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    print(f"Number of samples in training dataset (fold != {fold}): {len(df_train)}")
    print(f"Number of samples in validation dataset (fold == {fold}): {len(df_valid)}")

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    print(f"Number of samples in training dataset: {len(train_images)}")
    print(f"Number of samples in validation dataset: {len(valid_images)}")

    # Create and filter the training dataset
    train_dataset = ClassificationDataLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
        mean=mean,
        std=std
    )
    train_dataset = [entry for entry in train_dataset if entry is not None]

    # Check if there are samples in the training dataset
    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_bs,
            num_workers=2,
            shuffle=True
        )
    else:
        print("No samples in the training dataset.")
        train_loader = None

    # Create and filter the validation dataset
    valid_dataset = ClassificationDataLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
        mean=mean,
        std=std
    )
    valid_dataset = [entry for entry in valid_dataset if entry is not None]

    # Check if there are samples in the validation dataset
    if len(valid_dataset) > 0:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=valid_bs,
            shuffle=False,
            num_workers=2
        )
    else:
        print("No samples in the validation dataset.")
        valid_loader = None

    # Proceed with training only if there are samples in both datasets
    if train_loader is not None and valid_loader is not None:
        model = SEResNext50_32x4d(pretrained="imagenet")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            mode="max"
        )

        scaler = GradScaler()

        for epoch in range(epochs):
            print(f"Fold {fold}/{10}, Epoch {epoch + 1}/{epochs}")


            model.train()

            train_loader_iter = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}',total=len(train_loader))


            for batch in train_loader:

                image = batch["image"]
                target = batch["target"]
                image = image.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                optimizer.zero_grad()

                with autocast():
                    outputs, loss = model(image, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                torch.cuda.empty_cache()

                train_loader_iter.set_postfix(loss=loss.item())
                train_loader_iter.update()

            train_loader_iter.close()
            model.eval()

            predictions = []
            valid_loader_iter = tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', total=len(valid_loader))

            for batch in valid_loader:
                image = batch["image"]
                target = batch["target"]
                image = image.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                with autocast():
                    outputs, _ = model(image, target)

                predictions.append(outputs.sigmoid().detach().cpu().numpy())
                valid_loader_iter.update()

            valid_loader_iter.close()


            predictions = np.vstack(predictions).ravel()
            auc = metrics.roc_auc_score(valid_targets, predictions)
            scheduler.step(auc)
            print(f"AUC: {auc}")

            torch.cuda.empty_cache()


    else:
        print("Training or validation dataset is empty. Check your data paths.")
    

    print("Total time consumed -->",timer()-start) 

if __name__ == "__main__":
    for fold in range(10):
        train(fold)
        if model is not None:
            model_path_fold = os.path.join(model_path, f"model_fold_{fold}.pth")
            torch.save(model.state_dict(), model_path_fold)