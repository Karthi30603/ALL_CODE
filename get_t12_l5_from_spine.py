#!/usr/bin/env python3
"""
Extract T12 start and L5 end slice numbers from spine segmentation
"""

import sys
import os
import numpy as np
import nibabel as nib
from pathlib import Path

# Add VIBESegmentator to path
vibe_path = Path("/home/ai-user/VIBESegmentator")
sys.path.insert(0, str(vibe_path))

try:
    from TPTBox import Location, v_name2idx
    TPTBOX_AVAILABLE = True
except ImportError:
    TPTBOX_AVAILABLE = False

def analyze_spine_region():
    """Analyze spine region from semantic segmentation"""
    seg_path = "/home/ai-user/FAtseg/spine_semantic_seg.nii.gz"
    
    if not os.path.exists(seg_path):
        print(f"Error: {seg_path} not found")
        return None, None
    
    seg = nib.load(seg_path)
    data = seg.get_fdata().astype(int)
    
    # Label 49 is Vertebra_Corpus_border (general vertebra body)
    # Label 100 is Vertebra_Disc (intervertebral disc)
    vertebra_mask = (data == 49) | (data == 100)
    z_indices = np.where(vertebra_mask.any(axis=(1, 2)))[0]
    
    if len(z_indices) == 0:
        print("No spine region found in semantic segmentation")
        return None, None
    
    spine_start = int(z_indices.min())
    spine_end = int(z_indices.max())
    spine_length = len(z_indices)
    
    print("="*60)
    print("SPINE REGION FOUND:")
    print("="*60)
    print(f"Spine region spans slices: {spine_start} to {spine_end}")
    print(f"Total spine slices: {spine_length}")
    print()
    
    # Estimate T12 and L5 based on typical anatomy
    # T12 is typically at the upper portion of lumbar spine
    # L5 is at the bottom of lumbar spine
    # The visible spine region likely includes T12-L5
    
    # Conservative estimate: T12 starts near the beginning of visible spine
    # L5 ends near the end of visible spine
    t12_start = spine_start
    l5_end = spine_end
    
    # More refined: if we have a longer spine, T12 might be slightly lower
    # Typical lumbar spine (L1-L5) is about 5 vertebrae
    # If spine is very long, we might need to adjust
    if spine_length > 40:
        # Very long spine, T12 might be a bit lower
        t12_start = spine_start + max(0, (spine_length - 40) // 2)
    
    print("ESTIMATED VALUES:")
    print("="*60)
    print(f"T12 starting slice number: {t12_start}")
    print(f"L5 end slice number: {l5_end}")
    print()
    print("Note: These are estimates based on the visible spine region.")
    print("For precise identification, individual vertebra segmentation")
    print("would be needed (e.g., using SPINEPS).")
    print("="*60)
    
    return t12_start, l5_end

def main():
    t12_start, l5_end = analyze_spine_region()
    
    if t12_start is not None and l5_end is not None:
        # Save results
        results_file = "/home/ai-user/FAtseg/t12_l5_slices.txt"
        with open(results_file, "w") as f:
            f.write(f"T12_start_slice: {t12_start}\n")
            f.write(f"L5_end_slice: {l5_end}\n")
        print(f"\nResults saved to: {results_file}")
    else:
        print("\nCould not determine T12 and L5 slice numbers")

if __name__ == "__main__":
    main()



