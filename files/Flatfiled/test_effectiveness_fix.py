#!/usr/bin/env python3
"""
Test the fixed effectiveness analysis
"""

import numpy as np
import sys
import os
sys.path.append('/data/home/cmarc/SWIR_Projects/Flatfield')

from project_modules.FlatfieldProcessor import FlatfieldProcessor

def test_effectiveness_analysis():
    """Test the effectiveness analysis with sample data"""
    
    # Create sample data
    raw_image = np.random.randint(1000, 16000, size=(1024, 1280)).astype(np.float64)
    flatfield_map = np.ones_like(raw_image) * 0.98 + np.random.normal(0, 0.02, raw_image.shape)
    corrected_image = raw_image / flatfield_map
    
    # Create sample metadata
    metadata = {
        'original_mean': float(np.mean(raw_image)),
        'original_min': float(np.min(raw_image)),
        'original_max': float(np.max(raw_image)),
        'corrected_mean': float(np.mean(corrected_image)),
        'corrected_min': float(np.min(corrected_image)),
        'corrected_max': float(np.max(corrected_image)),
        'flatfield_mean': float(np.mean(flatfield_map)),
        'flatfield_min': float(np.min(flatfield_map)),
        'flatfield_max': float(np.max(flatfield_map)),
        'wheel_pos': 'test'
    }
    
    # Create processor and test the analysis
    processor = FlatfieldProcessor("pos1")
    
    try:
        print("Testing effectiveness analysis...")
        score, total, interpretation = processor._analyze_correction_effectiveness(
            raw_image, corrected_image, flatfield_map, metadata
        )
        print(f"\n✅ Test completed successfully!")
        print(f"Score: {score}/{total}, Interpretation: {interpretation}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_effectiveness_analysis()
    if success:
        print("\n🎉 The fix worked! The effectiveness analysis should now work properly.")
    else:
        print("\n❌ The fix didn't work. There may be other issues.")
