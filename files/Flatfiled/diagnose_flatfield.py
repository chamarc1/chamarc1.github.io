#!/usr/bin/env python3
"""
Diagnose Flatfield Correction Issues

This script helps diagnose why the flatfield correction is showing negative improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
from project_modules.FlatfieldProcessor import FlatfieldProcessor

def diagnose_flatfield_correction():
    """Diagnose the flatfield correction logic."""
    
    print("=" * 80)
    print("FLATFIELD CORRECTION DIAGNOSIS")
    print("=" * 80)
    
    # Test with pos1 first
    processor = FlatfieldProcessor("pos1")
    
    # Generate flatfield map
    print("Generating flatfield map for pos1...")
    try:
        flatfield_map = processor.generate_flatfield_map()
        
        if flatfield_map is None:
            print("❌ Failed to generate flatfield map")
            return
            
        print(f"✅ Flatfield map generated")
        print(f"   Shape: {flatfield_map.shape}")
        print(f"   Range: [{np.min(flatfield_map):.4f}, {np.max(flatfield_map):.4f}]")
        print(f"   Mean: {np.mean(flatfield_map):.4f}")
        print(f"   Optical center value: {flatfield_map[640, 510]:.4f}")
        
        # Get a sample image
        print("\nGetting sample image...")
        sample_images = processor.crossTrack_processor.generate_images(
            processor.cross_filter_pos, processor.cross_dark_pos
        )
        
        if not sample_images:
            print("❌ No sample images available")
            return
            
        sample_image = sample_images[0]["image"] if isinstance(sample_images[0], dict) else sample_images[0]
        print(f"✅ Sample image loaded")
        print(f"   Shape: {sample_image.shape}")
        print(f"   Range: [{np.min(sample_image):.1f}, {np.max(sample_image):.1f}]")
        print(f"   Mean: {np.mean(sample_image):.1f}")
        
        # Test both correction directions
        print("\n" + "=" * 60)
        print("TESTING CORRECTION DIRECTIONS")
        print("=" * 60)
        
        # Current method: divide by flatfield
        corrected_divide = sample_image.astype(np.float64) / flatfield_map
        
        # Alternative method: multiply by flatfield
        corrected_multiply = sample_image.astype(np.float64) * flatfield_map
        
        # Alternative method: use inverse of flatfield
        flatfield_inverted = 1.0 / flatfield_map
        corrected_inverted = sample_image.astype(np.float64) * flatfield_inverted
        
        print("1. CURRENT METHOD (divide by flatfield):")
        print(f"   Original uniformity: {calculate_uniformity(sample_image):.4f}")
        print(f"   Corrected uniformity: {calculate_uniformity(corrected_divide):.4f}")
        improvement_divide = ((calculate_uniformity(corrected_divide) - calculate_uniformity(sample_image)) / calculate_uniformity(sample_image)) * 100
        print(f"   Improvement: {improvement_divide:.2f}%")
        
        print("\n2. ALTERNATIVE METHOD (multiply by flatfield):")
        print(f"   Corrected uniformity: {calculate_uniformity(corrected_multiply):.4f}")
        improvement_multiply = ((calculate_uniformity(corrected_multiply) - calculate_uniformity(sample_image)) / calculate_uniformity(sample_image)) * 100
        print(f"   Improvement: {improvement_multiply:.2f}%")
        
        print("\n3. INVERTED METHOD (multiply by 1/flatfield):")
        print(f"   Corrected uniformity: {calculate_uniformity(corrected_inverted):.4f}")
        improvement_inverted = ((calculate_uniformity(corrected_inverted) - calculate_uniformity(sample_image)) / calculate_uniformity(sample_image)) * 100
        print(f"   Improvement: {improvement_inverted:.2f}%")
        
        # Create diagnostic plots
        create_diagnostic_plots(sample_image, flatfield_map, corrected_divide, corrected_multiply, corrected_inverted)
        
        # Recommend the best approach
        print("\n" + "=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        
        improvements = [improvement_divide, improvement_multiply, improvement_inverted]
        methods = ["Current (divide)", "Multiply", "Inverted"]
        
        best_idx = np.argmax(improvements)
        print(f"Best method: {methods[best_idx]} with {improvements[best_idx]:.2f}% improvement")
        
        if best_idx == 0:
            print("✅ Current method is working correctly!")
        elif best_idx == 1:
            print("⚠️  Should multiply by flatfield instead of dividing!")
        else:
            print("⚠️  Should use inverted flatfield (1/flatfield)!")
            
    except Exception as e:
        print(f"❌ Error during diagnosis: {str(e)}")
        raise

def calculate_uniformity(image):
    """Calculate uniformity as coefficient of variation (lower is more uniform)."""
    return np.std(image) / np.mean(image)

def create_diagnostic_plots(original, flatfield, corrected_div, corrected_mult, corrected_inv):
    """Create diagnostic plots showing different correction methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    im1 = axes[0, 0].imshow(original, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Original Image')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Flatfield map
    im2 = axes[0, 1].imshow(flatfield, cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Flatfield Map')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Current method (divide)
    im3 = axes[0, 2].imshow(corrected_div, cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Current Method (divide)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Alternative method (multiply)
    im4 = axes[1, 0].imshow(corrected_mult, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('Alternative (multiply)')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Inverted method
    im5 = axes[1, 1].imshow(corrected_inv, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('Inverted Method (1/flatfield)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Profile comparison
    row_idx = original.shape[0] // 2
    axes[1, 2].plot(original[row_idx, :], 'k-', label='Original', linewidth=2)
    axes[1, 2].plot(corrected_div[row_idx, :], 'r-', label='Current (divide)', linewidth=2)
    axes[1, 2].plot(corrected_mult[row_idx, :], 'g-', label='Multiply', linewidth=2)
    axes[1, 2].plot(corrected_inv[row_idx, :], 'b-', label='Inverted', linewidth=2)
    axes[1, 2].set_xlabel('Pixel Index')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].set_title('Cross-section Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Flatfield Correction Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = "/data/home/cmarc/SWIR_Projects/Flatfield/Images/flatfield_diagnosis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Diagnostic plots saved to: {save_path}")

if __name__ == "__main__":
    diagnose_flatfield_correction()
