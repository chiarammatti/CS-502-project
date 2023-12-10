import os
import pandas as pd
from google.cloud import storage
import tqdm

# Function to get a bucket
def get_bucket(bucket_name):
    storage_client = storage.Client(project="CS-502")
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

# Function to get blobs
def get_blobs(bucket_name, prefix):
    storage_client = storage.Client(project="CS-502")
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return bucket, blobs

# Read CSV file to get patient IDs
data = pd.read_csv("./final_500.csv")
patient_ids = data['patient_id'].unique()

patient_dir_base = './patient_data'

# Download data for each patient
for patient_id in tqdm.tqdm(patient_ids):
    patient_dir = os.path.join(patient_dir_base, str(patient_id))
    os.makedirs(patient_dir, exist_ok=True)

    temp_CT_folder = os.path.join(patient_dir, 'temp_CT')
    os.makedirs(temp_CT_folder, exist_ok=True)

    temp_seg_folder = os.path.join(patient_dir, 'temp_segs')
    os.makedirs(temp_seg_folder, exist_ok=True)

    # Download CT images
    bucket_nifti, blobs = get_blobs('rsna-competition-2023', 'train_nifti/' + str(patient_id))
    for blob in blobs:
        blob.download_to_filename(os.path.join(temp_CT_folder, blob.name.split('/')[-1]))

    # Download segmentation masks
    bucket_segs, blobs_segs = get_blobs('rsna-competition-2023', 'train_segmented/train_images/' + str(patient_id))
    for blob in blobs_segs:
        blob.download_to_filename(os.path.join(temp_seg_folder, blob.name.split('/')[-1]))
