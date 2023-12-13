# required package: pip install monai['all']
import torch
import torch.nn as nn
import os
import shutil
import tempfile
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
# from shutil import copyfile

# copyfile(src = "../input/monai-swunet/monai_swingunetr.py", dst = "../working/monai_swingunetr.py")
from swinUNETR_base import *



class ViT3DModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=14):
        super(ViT3DModel, self).__init__()
        model_dict = torch.load("./swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt")["state_dict"]
        self.base_model = SwinUNETR(
                img_size=(128,128,32),
                in_channels=in_channels,
                out_channels=num_classes,
                feature_size=48,
                use_checkpoint=False,
            )
        
          
        self.base_model.load_state_dict(model_dict, strict=False)

        # Option b
        for name, param in self.base_model.named_parameters():
            if not name.startswith("encoder1") and not name.startswith("encoder2"):
                param.requires_grad=False
        # Option a 
        for name, param in self.base_model.named_parameters():
            param.requires_grad=False
        
        
                
        
        # remove decoders
        del self.base_model.decoder1
        del self.base_model.decoder2
        del self.base_model.decoder3
        del self.base_model.decoder4
        del self.base_model.decoder5
        del self.base_model.out


        num_features = 4*4*768

        # Define output layers for each organ's classification
        self.fc_bowel = nn.Linear(num_features, 2)  # binary classification
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
        
        print(features['bowel'].shape)
        # Classification for each organ
        bowel = self.fc_bowel(features['bowel'].view(1,-1))
        liver = self.fc_liver(features['liver'].view(1,-1))
        kidney = self.fc_kidney(features['kidneys'].view(1,-1))
        spleen = self.fc_spleen(features['spleen'].view(1,-1))

        return bowel, liver, kidney, spleen

