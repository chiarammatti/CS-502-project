import pandas as pd
import torch
import torch.nn as nn
import torchio as tio
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
import logging

from dataset_helper import create_dataset 
from model import CNN3DModel, train_loop, val_loop

logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create dataloaders
data = pd.read_csv("./final_500_split.csv")

train_data = data[data['set'] == 'train']
val_data = data[data['set'] == 'val']
test_data = data[data['set'] == 'test']

training_transforms = tio.Compose([
    tio.ToCanonical(),
    tio.Resize((128, 128, -1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.CropOrPad(
    (128, 128, 32)),
])

validation_transforms = tio.Compose([
    tio.ToCanonical(),
    tio.Resize((128, 128, -1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.CropOrPad(
    (128, 128, 32)),
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
epochs = 1
best_val_loss = float('inf')

for epoch in tqdm.tqdm(range(epochs)):
    train_loss = train_loop(train_dataloader, model, criterion_binary, criterion_multiclass, optimizer)
    val_loss = val_loop(val_dataloader, model, criterion_binary, criterion_multiclass)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logging.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Save the model if validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')
        logging.info(f'Model saved at epoch {epoch+1}')

# Plotting
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

logging.info('Training completed')
