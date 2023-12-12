import os
import pandas as pd
import torchio as tio
import tqdm
import nibabel as nib
import numpy as np
import torch

def combine_masks_and_create_labelmap(path_list):
    combined_mask = None

    for path in path_list:
        mask = nib.load(path).get_fdata()
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = np.maximum(combined_mask, mask)
    combined_mask_4d = combined_mask[np.newaxis, ...]
    combined_mask_tensor = torch.tensor(combined_mask_4d, dtype=torch.float32)
    return combined_mask_tensor


def create_dataset(data_subset, patient_dir_base, transforms):
    subjects = []
    for index, row in tqdm.tqdm(data_subset.iterrows(), total=data_subset.shape[0]):
        patient_id = row['patient_id']
        patient_dir = os.path.join(patient_dir_base, str(patient_id))
        temp_CT_folder = os.path.join(patient_dir, 'temp_CT')
        temp_seg_folder = os.path.join(patient_dir, 'temp_segs')

        # Assuming the CT image is the first file in the temp_CT_folder
        ct_files = os.listdir(temp_CT_folder)
        if not ct_files:
            continue  # Skip if no CT files found
        ct_path = os.path.join(temp_CT_folder, ct_files[0])
        ct_image = tio.ScalarImage(ct_path)
        resample_transform = tio.Resample(ct_image)

        # Combine kidney masks
        kidney_paths = [
            os.path.join(temp_seg_folder, "kidney_left.nii.gz"),
            os.path.join(temp_seg_folder, "kidney_right.nii.gz")
        ]
        combined_kidney_mask = combine_masks_and_create_labelmap(kidney_paths)

        # Combine bowel masks
        bowel_paths = [
            os.path.join(temp_seg_folder, "duodenum.nii.gz"),
            os.path.join(temp_seg_folder, "colon.nii.gz"),
            os.path.join(temp_seg_folder, "small_bowel.nii.gz"),
            os.path.join(temp_seg_folder, "esophagus.nii.gz")
        ]
        combined_bowel_mask = combine_masks_and_create_labelmap(bowel_paths)

        # Create a tio.Subject
        subject = tio.Subject(
            CT=tio.ScalarImage(ct_path),
            segmentation_kidneys=resample_transform(tio.LabelMap(tensor=combined_kidney_mask)),
            segmentation_liver=resample_transform(tio.LabelMap(os.path.join(temp_seg_folder, "liver.nii.gz"))),
            segmentation_spleen=resample_transform(tio.LabelMap(os.path.join(temp_seg_folder, "spleen.nii.gz"))),
            segmentation_bowel=resample_transform(tio.LabelMap(tensor=combined_bowel_mask)),
            patient_id=patient_id,
        )
        subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=transforms)
