import pandas as pd
import torch
import torch.nn as nn
import torchio as tio
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt

from dataset_helper import create_dataset 
from model import CNN3DModel, train_loop, val_loop


# Create dataloaders
data = pd.read_csv("./final_500_splits.csv")

train_data = data[data['set'] == 'train']
val_data = data[data['set'] == 'val']
test_data = data[data['set'] == 'test']

train_dataset = create_dataset(train_data, './patient_data')
val_dataset = create_dataset(val_data, './patient_data')
test_dataset = create_dataset(test_data, './patient_data')

training_transforms = tio.Compose([
    tio.ToCanonical(),
    tio.Resize((128, 128, -1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

validation_transforms = tio.Compose([
    tio.ToCanonical(),
    tio.Resize((128, 128, -1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

train_dataloader = create_dataset(data[data['set'] == 'train'], './patient_data', training_transforms)
val_dataloader = create_dataset(data[data['set'] == 'val'], './patient_data', validation_transforms)
test_dataloader = create_dataset(data[data['set'] == 'test'], './patient_data', validation_transforms)

print('Train dataset size:', len(train_dataloader), 'subjects')
print('Val dataset size:', len(val_dataloader), 'subjects')
print('Test dataset size:', len(test_dataloader), 'subjects')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN3DModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_multiclass = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
epochs = 50

for epoch in tqdm.tqdm(range(epochs)):
    train_loss = train_loop(train_dataloader, model, criterion_binary, criterion_multiclass, optimizer)
    val_loss = val_loop(val_dataloader, model, criterion_binary, criterion_multiclass)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}")


plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()