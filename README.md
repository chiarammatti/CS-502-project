# CS-502-project: AI-Based Abdominal Trauma Detection

This is the project repository of Group 3 for EPFL's course Deep Learning in Biomedicine (CS-502). It contains all the code to develop a classifier to accurately detect and classify the type and severity of injuries within the abdomen based on CT scans. This project was inspired by the [RSNA 2023 Abdominal Trauma Detection Challenge](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/)

## Data Loading 
The dataset used was sourced from the RSNA 2023 Abdominal Trauma Detection Challenge and uploaded from Kaggle to Google Cloud Storage (GCS) using the provided `fix_and_load_data.py` script. To avoid future errors, during the uploading process the mandatory "Modality" attribute of DICOM files was added, as many of the provided images where missing it.  
Subsequently, to be able to work with 3D models, all the DICOM files were converted into NIfTI format and uploaded back to Google Cloud Storage, using `nifti_conversion.py`.
To direcly have access to GCS bucket please follow [this link](https://console.cloud.google.com/storage/browser/rsna-competition-2023). 

## Usage

#### Training 
- Run `save_data.py` to download the data from GCS locally. Note that only the subjects specified in CSV file `final_500_split.csv` will be downloaded (CT scans and corresponding segmentations)  
- Run `train_model.py`
  
Note: The models used are defined in `model.py`. SwinUNETR pretrained weights can be downloaded [here](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt).

#### Testing
- Run `save_data.py` to download the data from GCS locally. Note that only the subjects specified in CSV file `final_500_split.csv` will be downloaded (CT scans and corresponding segmentations)  
- Run `inference.py` to evaluate the model





