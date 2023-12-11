import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from google.cloud import storage
import pydicom
import itertools
import shutil
from totalsegmentator.python_api import totalsegmentator # after installation of git+https://github.com/wasserth/TotalSegmentator.git

# Function to get a bucket
def get_bucket(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client(project = "CS-502")
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

# Function to get blobs
def get_blobs(bucket_name, prefix):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client(project = "CS-502")
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return bucket, blobs

 # Function to segment the images
def segment_imgs(bucket, blobs):
    """
    bucket: bucket object
    blobs: list of blobs for one subject and run
    
    """
    errors = [] 
    
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
        if not bucket.blob(os.path.join('train_segmented', fn, 'stomach.nii.gz')).exists():
            print(fn)
            if blob.name.endswith('.dcm'):
                blob.download_to_filename(os.path.join(temp_source_folder, blob.name.split('/')[-1]))
    
    try:
        # segment the files
        totalsegmentator(temp_source_folder, temp_result_folder, roi_subset = ["spleen", "kidney_left", "kidney_right", "liver", "esophagus", "colon", "duodenum", "small_bowel", "stomach"])
    except:
        errors.append(fn)
    
    # upload the files
    for i, filename in enumerate(os.listdir(temp_result_folder)):
        output_path = os.path.join('train_segmented', fn, filename)
        bucket.blob(output_path).upload_from_filename(os.path.join(temp_result_folder, filename))
    
    # delete temp files
    shutil.rmtree(temp_source_folder)
    shutil.rmtree(temp_result_folder)
    
    return errors
    
bucket, all_blobs = get_blobs('rsna-competition-2023', 'train_images_fixed/train_images')
blobs_sep = [list(blobs) for _, blobs in itertools.groupby(all_blobs, lambda blob: blob.name.split('/')[2:4])]
all_errors = []
for run in blobs_sep:
    print(len(run))
    err = segment_imgs(run)
    all_errors.append(err)
    print(' ################################# Run done: ', run[0].name.split('/')[2:4], '#################################')