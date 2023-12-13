import numpy as np 
import pandas as pd 
import os
from google.cloud import storage
import dicom2nifti
from glob import glob
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import torch
import torchio as tio
import torchvision
import skimage
from skimage import color
from torch.utils.data import DataLoader
import shutil

# load data from google cloud storage
def get_blobs(bucket_name, prefix):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client(project = "CS-502")
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return bucket, blobs

def get_bucket(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client(project = "CS-502")
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

def get_directories(bucket_name, prefix):
    """Get a list of directories from Google Cloud Storage blobs."""
    
    
    bucket, blobs = get_blobs(bucket_name, prefix)
    

    subdirectories = []
    subjects = []

    for blob in blobs:
        # Split the blob name into parts using '/' as the separator
        parts = blob.name.split('/')

        # If there are more than one part, and the first part is the specified directory
        if len(parts)>1 and parts[-2] not in subdirectories:
            # Add the subdirectory to the set
            subdirectories.append(parts[-2])
            subjects.append(parts[-3])
    return subdirectories, subjects


import shutil
def dcm2nifti(blobs, file_error, subject, direct, nifti_list):
    """
    blobs: list of blobs for one subject and run
    
    """
    bucket = get_bucket('cs-502-project')

    # temporary folder to save the files
    temp_source_folder = 'temp_source'
    if not os.path.exists(temp_source_folder):
        os.makedirs(temp_source_folder)

    temp_result_folder = 'temp_result'
    if not os.path.exists(temp_result_folder):
        os.makedirs(temp_result_folder)
    
    # download the files

    for blob in blobs:
        fn = '/'.join(blob.name.split('/')[1:-1])
        if not bucket.blob(os.path.join('train_nifti', fn + '.nii.gz')).exists():
            if blob.name.endswith('.dcm'):
                blob.download_to_filename(os.path.join(temp_source_folder, blob.name.split('/')[-1]))
                
    try:
        dicom2nifti.dicom_series_to_nifti(temp_source_folder, os.path.join(temp_result_folder, subject + "_" + direct + ".nii.gz"))
    except dicom2nifti.exceptions.ConversionValidationError:
        file_error.append(sub + "_" + direct)
        pass
    
    
    # upload the files
    output_path = os.path.join('train_nifti/' + subject + "/" + direct + ".nii.gz")
    bucket.blob(output_path).upload_from_filename(os.path.join(temp_result_folder, subject + "_" + direct + ".nii.gz"))
        
    shutil.rmtree(temp_source_folder)
    shutil.rmtree(temp_result_folder)
    
    nifti_list.append(output_path)
    
        
    return file_error, nifti_list 


dir_fixed, subs = get_directories("rsna-competition-2023", "train_images_fixed/train_images")
ka = 0
for sub, direct in zip(subs, dir_fixed):
    bucket, all_blobs = get_blobs('rsna-competition-2023', 'train_images_fixed/train_images/' + sub + "/" + direct)
    
    file_err, nifti_list = dcm2nifti(all_blobs, file_err, sub, direct, nifti_list)
    ka = ka + 1
    
    if ka % 100 == 0:
        print("files saved:", ka)

file_path = "file_errorss.txt"
# Writing to a text file
with open(file_path, mode='w') as file:
    for item in file_err:
        file.write(f"{item}\n")
        
df_nifti = pd.DataFrame(nifti_list)
df_nifti.to_csv( "./nifti_list.csv", index = False, encoding = 'utf-8-sig')