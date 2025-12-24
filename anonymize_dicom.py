#!/usr/bin/env python3
"""
Script to anonymize DICOM files by removing patient identifying information.
"""

import os
import sys
from pathlib import Path
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

# Tags to anonymize (set to "unknown")
ANONYMIZE_TAGS = {
    # Patient identification
    (0x0010, 0x0010): 'unknown',  # Patient's Name
    (0x0010, 0x0020): 'unknown',  # Patient ID
    (0x0010, 0x0021): 'unknown',  # Issuer of Patient ID
    (0x0010, 0x0030): 'unknown',  # Patient's Birth Date
    (0x0010, 0x0032): 'unknown',  # Patient's Birth Time
    (0x0010, 0x0040): 'unknown',  # Patient's Sex
    (0x0010, 0x1000): 'unknown',  # Other Patient IDs
    (0x0010, 0x1001): 'unknown',  # Other Patient Names
    (0x0010, 0x1005): 'unknown',  # Patient's Birth Name
    (0x0010, 0x1010): 'unknown',  # Patient's Age
    (0x0010, 0x1040): 'unknown',  # Patient's Address
    (0x0010, 0x1060): 'unknown',  # Patient's Mother's Birth Name
    (0x0010, 0x2150): 'unknown',  # Patient's Telecom Information
    (0x0010, 0x2152): 'unknown',  # Patient's Telecom Information
    (0x0010, 0x2154): 'unknown',  # Patient's Telecom Information
    (0x0010, 0x2160): 'unknown',  # Ethnic Group
    (0x0010, 0x4000): 'unknown',  # Patient Comments
    
    # Institution and equipment
    (0x0008, 0x0080): 'unknown',  # Institution Name
    (0x0008, 0x0081): 'unknown',  # Institution Address
    (0x0008, 0x0090): 'unknown',  # Referring Physician's Name
    (0x0008, 0x0092): 'unknown',  # Referring Physician's Address
    (0x0008, 0x0094): 'unknown',  # Referring Physician's Telephone Numbers
    (0x0008, 0x1010): 'unknown',  # Station Name
    (0x0008, 0x1040): 'unknown',  # Institutional Department Name
    (0x0008, 0x1048): 'unknown',  # Physician(s) of Record
    (0x0008, 0x1049): 'unknown',  # Physician(s) of Record Identification Sequence
    (0x0008, 0x1050): 'unknown',  # Performing Physician's Name
    (0x0008, 0x1052): 'unknown',  # Performing Physician Identification Sequence
    (0x0008, 0x1060): 'unknown',  # Name of Physician(s) Reading Study
    (0x0008, 0x1062): 'unknown',  # Physician(s) Reading Study Identification Sequence
    (0x0008, 0x1070): 'unknown',  # Operator's Name
    (0x0008, 0x1072): 'unknown',  # Operator Identification Sequence
    
    # Study and Series
    (0x0020, 0x0010): 'unknown',  # Study ID
    (0x0032, 0x1032): 'unknown',  # Requesting Physician
    (0x0032, 0x1060): 'unknown',  # Requested Procedure Description
    (0x0040, 0x0244): 'unknown',  # Performed Procedure Step Start Date
    (0x0040, 0x0245): 'unknown',  # Performed Procedure Step Start Time
    (0x0040, 0x0253): 'unknown',  # Performed Procedure Step ID
    (0x0040, 0x0254): 'unknown',  # Performed Procedure Step Description
    (0x0040, 0x0275): 'unknown',  # Request Attributes Sequence
    (0x0040, 0x1001): 'unknown',  # Requested Procedure ID
    (0x0040, 0x1004): 'unknown',  # Requested Procedure Code Sequence
    (0x0040, 0x1400): 'unknown',  # Requested Procedure Comments
}

def anonymize_dicom_file(file_path):
    """Anonymize a single DICOM file."""
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(file_path)
        
        # Anonymize tags
        for tag, value in ANONYMIZE_TAGS.items():
            if tag in ds:
                if isinstance(ds[tag].value, (str, int, float)):
                    ds[tag].value = value
                elif isinstance(ds[tag].value, Sequence):
                    # For sequences, we can clear them or leave empty
                    ds[tag].value = Sequence()
        
        # Explicitly set Patient Name to unknown if it exists
        if hasattr(ds, 'PatientName') or (0x0010, 0x0010) in ds:
            ds.PatientName = 'unknown'
        
        # Save the anonymized file
        ds.save_as(file_path, write_like_original=False)
        return True
    except Exception as e:
        print(f"Error anonymizing {file_path}: {e}", file=sys.stderr)
        return False

def anonymize_directory(directory):
    """Recursively anonymize all DICOM files in a directory."""
    directory = Path(directory)
    dicom_files = []
    
    # Find all DICOM files (excluding DICOMDIR)
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.name != 'DICOMDIR':
            try:
                # Try to read as DICOM to verify it's a DICOM file
                pydicom.dcmread(file_path, stop_before_pixels=True)
                dicom_files.append(file_path)
            except:
                # Not a DICOM file, skip
                pass
    
    print(f"Found {len(dicom_files)} DICOM files to anonymize")
    
    # Anonymize each file
    success_count = 0
    for file_path in dicom_files:
        if anonymize_dicom_file(file_path):
            success_count += 1
        if success_count % 10 == 0:
            print(f"Anonymized {success_count}/{len(dicom_files)} files...")
    
    print(f"Successfully anonymized {success_count}/{len(dicom_files)} DICOM files")
    return success_count == len(dicom_files)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 anonymize_dicom.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    
    success = anonymize_directory(directory)
    sys.exit(0 if success else 1)

