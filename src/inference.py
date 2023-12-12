import pandas as pd
import torch
import torchio as tio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset_helper import create_dataset
from model import CNN3DModel, get_labels, label_dict

def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model


data = pd.read_csv("final_500_split.csv")

val_data = data[data['set'] == 'val']
test_data = data[data['set'] == 'test']


validation_transforms = tio.Compose([
    tio.ToCanonical(),
    tio.Resize((128, 128, -1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

val_dataloader = create_dataset(data[data['set'] == 'val'], './patient_data', validation_transforms)
test_dataloader = create_dataset(data[data['set'] == 'test'], './patient_data', validation_transforms)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN3DModel().to(device)
model = load_checkpoint(model, 'checkpoints_lr0001/model_epoch_39.pth')

def run_inference_and_calculate_metrics(dataloader, model, device):
    metrics = {'bowel': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
               'liver': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
               'kidney': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
               'spleen': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}}
    
    total_batches = len(dataloader)
    for batch in dataloader:
        inputs = batch['CT']['data'].to(device)
        masks = {
            'bowel': batch['segmentation_bowel']['data'].cuda(),
            'liver': batch['segmentation_liver']['data'].cuda(),
            'kidneys': batch['segmentation_kidneys']['data'].cuda(),
            'spleen': batch['segmentation_spleen']['data'].cuda()
        }
        labels = get_labels(batch, label_dict)
        outputs = model(inputs, masks)

        for i,organ in enumerate(['bowel', 'liver', 'kidney', 'spleen']):
            predictions = torch.sigmoid(outputs[i]).squeeze().cpu().detach().numpy()
            predictions = (predictions > 0.5).astype(int)
           

            organ_labels = labels[organ].cpu().numpy()
            organ_labels = organ_labels[0]

            # Update metrics for each organ
            metrics[organ]['accuracy'] += round(accuracy_score(organ_labels, predictions), 3)
            metrics[organ]['precision'] += round(precision_score(organ_labels, predictions, zero_division=0), 3)
            metrics[organ]['recall'] += round(recall_score(organ_labels, predictions, zero_division=0), 3)
            metrics[organ]['f1_score'] += round(f1_score(organ_labels, predictions, zero_division=0), 3)

    # Average the metrics over all batches
    for organ in metrics:
        for metric in metrics[organ]:
            metrics[organ][metric] /= total_batches

    return metrics

# Example usage:
val_metrics = run_inference_and_calculate_metrics(val_dataloader, model, device)
print(f"Validation Metrics: {val_metrics}")

test_metrics = run_inference_and_calculate_metrics(test_dataloader, model, device)
print(f"Test Metrics: {test_metrics}")
