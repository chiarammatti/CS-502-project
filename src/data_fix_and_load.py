import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from google.cloud import storage
import pydicom
import itertools

# Function to add missing attributes to dicom files
def add_missing_attributes(data_path, attributes_dict):
    """
    data_path: path where files to modify are (train_images/subject_ID/run_ID/image_ID.dcm)
    attribute_ids: dict of attributes that need to be added, e.g. attribute[0x0080060] = ('CS', 'CT')
    """
    
    dcm = pydicom.dcmread(data_path)
    for tag, (VR, value) in attributes_dict.items():
        if not tag in dcm:
            dcm.add_new(tag, VR, value)
            dcm.save_as(data_path)

# Function to get the blobs from the bucket
def get_blobs(bucket_name, prefix):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return bucket, blobs

# Get the blobs
bucket, blobs = get_blobs('rsna-competition-2023', 'train_images')
blobs_sep = [list(blobs) for _, blobs in itertools.groupby(blobs, lambda blob: blob.name.split('/')[1:3])] 

# Define missing attributes
attributes_dict = {0x0080060 : ('CS', 'CT')}

for run in blobs_sep:
    for blob in run[::10]:
        # check output file exists
        output_path = os.path.join('train_images_fix', blob.name)
        if not bucket.blob(output_path).exists():
            if blob.name.endswith('.dcm'):
                # get the file from the bucket
                blob.download_to_filename('temp.dcm')
                # add the missing attributes
                add_missing_attributes('temp.dcm', attributes_dict)        
                # save in output folder
                output_path = os.path.join('train_images_fix', blob.name)
                bucket.blob(output_path).upload_from_filename('temp.dcm')
                # delete temp file
                os.remove('temp.dcm')
