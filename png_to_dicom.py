#!/usr/bin/env python3
import os
import glob
from datetime import datetime
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

def png_to_dicom(png_path, output_dir, patient_name="Pavithra Tallam", instance_number=1, 
                 study_instance_uid=None, series_instance_uid=None, series_date=None, series_time=None):
    """
    Convert PNG image to DICOM format with specified metadata.
    """
    # Read PNG image
    img = Image.open(png_path)
    img_array = np.array(img.convert('L'))  # Convert to grayscale
    
    # Get filename without extension for DICOM filename
    base_name = os.path.splitext(os.path.basename(png_path))[0]
    dicom_path = os.path.join(output_dir, f"{base_name}.dcm")
    
    # Try to extract instance number from filename (e.g., "Set 1_000" -> 0, "Set 1_001" -> 1)
    try:
        # Extract number after underscore (e.g., "Set 1_001" -> "001")
        parts = base_name.split('_')
        if len(parts) > 1:
            instance_num = int(parts[-1])
        else:
            instance_num = instance_number
    except:
        instance_num = instance_number
    
    # Create DICOM dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
    file_meta.ImplementationClassUID = generate_uid()
    
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    
    # Set required DICOM tags
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
    # Patient information
    ds.PatientName = patient_name
    ds.PatientID = "UNKNOWN"
    # PatientBirthDate - leave empty or use valid date format
    ds.PatientSex = "O"  # Other/Unknown
    ds.PatientAge = "000Y"  # Unknown age format
    
    # Study information - use shared UID if provided
    if study_instance_uid is None:
        study_instance_uid = generate_uid()
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyID = "UNKNOWN"
    ds.StudyDescription = "UNKNOWN"
    ds.AccessionNumber = ""
    
    # Series information - use shared UID if provided
    if series_instance_uid is None:
        series_instance_uid = generate_uid()
    ds.SeriesInstanceUID = series_instance_uid
    ds.SeriesNumber = 1
    ds.SeriesDescription = "UNKNOWN"
    ds.Modality = "MR"  # MRI
    if series_date is None:
        series_date = datetime.now().strftime("%Y%m%d")
    if series_time is None:
        series_time = datetime.now().strftime("%H%M%S")
    ds.SeriesDate = series_date
    ds.SeriesTime = series_time
    
    # Image information
    ds.InstanceNumber = instance_num
    ds.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = img_array.shape[0]
    ds.Columns = img_array.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]
    # SliceThickness - optional, can be empty
    # SpacingBetweenSlices - optional, can be empty
    
    # Equipment information
    ds.Manufacturer = "UNKNOWN"
    ds.ManufacturerModelName = "UNKNOWN"
    ds.DeviceSerialNumber = ""
    ds.SoftwareVersions = "UNKNOWN"
    
    # Set pixel data
    # Convert to uint16 and scale if needed
    if img_array.dtype != np.uint16:
        # Scale to 16-bit range
        img_array = (img_array.astype(np.float32) / img_array.max() * 65535).astype(np.uint16)
    
    ds.PixelData = img_array.tobytes()
    
    # Save DICOM file
    ds.save_as(dicom_path, write_like_original=False)
    print(f"Converted: {os.path.basename(png_path)} -> {os.path.basename(dicom_path)}")
    
    return dicom_path

def convert_directory(png_dir, output_dir=None, patient_name="Pavithra Tallam"):
    """
    Convert all PNG files in a directory to DICOM format.
    """
    if output_dir is None:
        output_dir = os.path.join(png_dir, "DICOM")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PNG files
    png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    
    if not png_files:
        print(f"No PNG files found in {png_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to convert...")
    
    # Generate shared UIDs for all files in this series
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    series_date = datetime.now().strftime("%Y%m%d")
    series_time = datetime.now().strftime("%H%M%S")
    
    print(f"Using shared SeriesInstanceUID: {series_instance_uid}")
    
    for idx, png_path in enumerate(png_files, start=1):
        try:
            png_to_dicom(png_path, output_dir, patient_name, instance_number=idx,
                        study_instance_uid=study_instance_uid,
                        series_instance_uid=series_instance_uid,
                        series_date=series_date,
                        series_time=series_time)
        except Exception as e:
            print(f"Error converting {os.path.basename(png_path)}: {e}")
    
    print(f"\nConversion complete! DICOM files saved to: {output_dir}")

if __name__ == "__main__":
    png_directory = "/home/ai-user/PNG/MRI MSK-SHOULDER - Set 5_000"
    patient_name = "Pavithra Tallam"
    
    convert_directory(png_directory, patient_name=patient_name)

