import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import pandas as pd 

data = pd.read_csv("./final_500_splits.csv")
label_dict = data.set_index('patient_id').to_dict(orient='index')



class CNN3DModel(nn.Module):
    def __init__(self):
        super(CNN3DModel, self).__init__()
        #load a pretrained 3D ResNet and replace the final fully connected layer        
        self.base_model = r3d_18(pretrained=True)

        # Modify the first convolution layer for single-channel input
        original_first_layer = self.base_model.stem[0]
        self.base_model.stem[0] = nn.Conv3d(
            in_channels=1, 
            out_channels=original_first_layer.out_channels, 
            kernel_size=original_first_layer.kernel_size, 
            stride=original_first_layer.stride, 
            padding=original_first_layer.padding, 
            bias=False
        )
        with torch.no_grad():
            self.base_model.stem[0].weight = nn.Parameter(
                original_first_layer.weight.mean(dim=1, keepdim=True)
            )
            
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        # Define output layers for each organ's classification
        self.fc_bowel = nn.Linear(num_features, 2)  # binary classification
        #self.fc_extravasation = nn.Linear(num_features, 2) 
        self.fc_liver = nn.Linear(num_features, 3)  # multiclass classification
        self.fc_kidney = nn.Linear(num_features, 3)
        self.fc_spleen = nn.Linear(num_features, 3)

    def apply_mask(self, inputs, mask):
        return inputs * mask.unsqueeze(1)  # Adding channel dimension to mask

    def forward(self, x, masks):
        # Apply masks and extract features for each organ
        features = {}
        for organ, mask in masks.items():
            masked_input = self.apply_mask(x, mask)
            features[organ] = self.base_model(masked_input)

        # Classification for each organ
        bowel = self.fc_bowel(features['bowel'])
        #extravasation = self.fc_extravasation(features['extravasation'])
        liver = self.fc_liver(features['liver'])
        kidney = self.fc_kidney(features['kidneys'])
        spleen = self.fc_spleen(features['spleen'])

        return bowel, liver, kidney, spleen #extravasation, 

def calculate_loss(outputs, labels, criterion_binary, criterion_multiclass):
    outputs_bowel, outputs_liver, outputs_kidneys, outputs_spleen = outputs
    loss_bowel = criterion_binary(outputs_bowel, labels['bowel'])
    loss_liver = criterion_multiclass(outputs_liver, labels['liver'].argmax(dim=1))
    loss_kidney = criterion_multiclass(outputs_kidneys, labels['kidney'].argmax(dim=1))
    loss_spleen = criterion_multiclass(outputs_spleen, labels['spleen'].argmax(dim=1))

    total_loss = loss_bowel + loss_liver + loss_kidney + loss_spleen
    return total_loss


def train_loop(dataloader, model, criterion_binary, criterion_multiclass, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch['CT']['data'].cuda()
        masks = {
            'bowel': batch['segmentation_bowel']['data'].cuda(),
            'liver': batch['segmentation_liver']['data'].cuda(),
            'kidneys': batch['segmentation_kidneys']['data'].cuda(),
            'spleen': batch['segmentation_spleen']['data'].cuda()
        }
        labels = get_labels(batch, label_dict) 
        optimizer.zero_grad()
        outputs = model(inputs, masks)
        loss = calculate_loss(outputs, labels, criterion_binary, criterion_multiclass)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def val_loop(dataloader, model, criterion_binary, criterion_multiclass):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['CT']['data'].cuda()
            masks = {
                'bowel': batch['segmentation_bowel']['data'].cuda(),
                'liver': batch['segmentation_liver']['data'].cuda(),
                'kidneys': batch['segmentation_kidneys']['data'].cuda(),
                'spleen': batch['segmentation_spleen']['data'].cuda()
            }
            labels = get_labels(batch, label_dict) 
            outputs = model(inputs, masks)
            loss = calculate_loss(outputs, labels, criterion_binary, criterion_multiclass)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss



def get_labels(batch, label_dict):
    labels = {'bowel': [], 'liver': [], 'kidney': [], 'spleen': []} #'extravasation': [],
    pid = batch['patient_id']
    patient_labels = label_dict[int(pid)]
    labels['bowel'].append([patient_labels['bowel_healthy'], patient_labels['bowel_injury']])
    #labels['extravasation'].append([patient_labels['extravasation_healthy'], patient_labels['extravasation_injury']])
    labels['liver'].append([patient_labels['liver_healthy'], patient_labels['liver_low'], patient_labels['liver_high']])
    labels['kidney'].append([patient_labels['kidney_healthy'], patient_labels['kidney_low'], patient_labels['kidney_high']])
    labels['spleen'].append([patient_labels['spleen_healthy'], patient_labels['spleen_low'], patient_labels['spleen_high']])
    return {k: torch.tensor(v, dtype=torch.float32).cuda() for k, v in labels.items()}