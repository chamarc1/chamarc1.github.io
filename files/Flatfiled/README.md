# SWIR Flatfield Correction Pipeline

**Enhanced Position-Optimized Illumination Uniformity Correction for AirHARP2 SWIR Instrument**

This project provides a comprehensive Python-based pipeline for processing Short-Wave Infrared (SWIR) images to generate high-quality flatfield corrections. The pipeline implements advanced quadratic envelope fitting techniques with position-specific optimization to achieve maximum illumination uniformity across all filter positions.

## 🚀 What This Tool Does

The enhanced pipeline automatically processes SWIR images from all four filter wheel positions and creates position-optimized flatfield corrections that remove spatial variations in sensor response. Key features include:

- **Position-optimized correction methods** for each filter wavelength
- **Automated strategy selection** (multiply/divide/hybrid) based on filter position
- **Comprehensive validation framework** with real-time quality metrics
- **1.5-2% uniformity improvement** achieved for most filter positions
- **Complete diagnostic and analysis suite** for scientific documentation

## 📊 Performance Summary

| Position | Wavelength | Method | Mean Improvement | Rating |
|----------|------------|--------|------------------|--------|
| pos1 | 1.57μm | multiply | 1.62% | ✅ GOOD |
| pos2 | 1.55μm | multiply | 1.98% | ✅ GOOD (Best) |
| pos3 | 1.38μm | divide | 0.31% | ⚠️ MODERATE |
| pos4 | Open | multiply | 1.63% | ✅ GOOD |

## 🏗️ Project Structure

### Core Processing Modules

* **`Main.py`**: Entry point that processes all filter positions automatically. Run this script to generate flatfield corrections for all positions with comprehensive PowerPoint-ready visualizations.

* **`project_modules/FlatfieldProcessor.py`**: Enhanced processor with:
    - Position-optimized correction logic (`auto`, `multiply`, `divide`, `hybrid`)
    - Advanced profile extraction with sigma filtering
    - Quadratic envelope fitting and 2D map generation
    - Comprehensive validation and quality metrics
    - PowerPoint-ready plot generation

* **`project_modules/CompositeProcessor.py`**: Handles image data management:
    - Loading and organizing images by filter position
    - Dark frame correction and averaging
    - Metadata association and validation

* **`project_modules/ImageProcessor.py`**: Low-level image operations and utilities

* **`project_modules/Constants.py`**: System configuration and filter mappings

### Analysis and Diagnostic Tools

* **`create_summary_visualization.py`**: Generates comprehensive performance analysis and comparison tables

* **`analyze_flatfield_effectiveness.py`**: Detailed effectiveness analysis with method comparison

* **`diagnose_flatfield.py`**: Diagnostic tool for correction strategy evaluation

* **`test_updated_profiles.py`**: Validation testing with position-optimized methods

## ⚡ Quick Start

### Basic Usage - Process All Positions

```bash
# Process all filter positions with default settings
python Main.py

# Adjust smoothing for more/less aggressive correction
python Main.py --num_sigma 2.0    # More smoothing
python Main.py --num_sigma 0.5    # Less smoothing
```

### Advanced Usage - Analysis and Validation

```bash
# Generate comprehensive performance analysis
python create_summary_visualization.py

# Run position-optimized validation tests
python test_updated_profiles.py

# Perform diagnostic analysis
python analyze_flatfield_effectiveness.py
```

## 🎯 Key Features

### Position-Optimized Correction

The pipeline automatically selects the optimal correction method for each filter position:

- **pos1, pos2, pos4**: Multiply correction (optimal for these wavelengths)
- **pos3**: Divide correction (specifically optimized for 1.38μm characteristics)

### Comprehensive Validation

- **Real-time quality metrics** during processing
- **Uniformity improvement quantification** (% reduction in standard deviation)
- **Center-to-edge analysis** for spatial uniformity assessment
- **Statistical validation** with multiple image samples

### Advanced Diagnostics

- **Strategy comparison analysis** across multiple correction methods
- **Regional uniformity assessment** for different detector areas
- **Spatial frequency analysis** for correction effectiveness evaluation

## 📈 Output Files and Visualizations

### PowerPoint-Ready Plots (`Images/PowerPoint_Plots/`)

- **Profile Analysis** (`slide4_profile_analysis_pos[1-4].png`): Raw data, filtering, smoothing, and envelope fitting
- **3D Envelope Plots** (`slide5_3d_envelope_pos[1-4].png`): Spatial visualization of illumination patterns
- **Flatfield Maps** (`slide6_flatfield_map_pos[1-4].png`): 2D correction maps with cross-sections
- **Before/After Comparison** (`slide7_before_after_comparison_pos[1-4].png`): Correction effectiveness demonstration
- **Summary Analysis** (`slide8_*.png`): Multi-position comparison and coefficient analysis

### Performance Analysis (`Images/`)

- **Performance Summary** (`flatfield_performance_summary.png`): Comprehensive results overview
- **Strategy Comparison** (`strategy_comparison_pos[1-4].png`): Method effectiveness analysis
- **Regional Analysis** (`regional_analysis_pos[1-4].png`): Spatial uniformity assessment

### Validation Results

- **Position-optimized test results** (`updated_test_results_pos[1-4].json`)
- **Combined summary** (`position_optimized_summary.json`)

## 🔧 Advanced Configuration

### Applying Corrections to Science Data

```python
from project_modules.FlatfieldProcessor import FlatfieldProcessor

# Initialize processor for specific position
processor = FlatfieldProcessor("pos1")

# Apply position-optimized correction
corrected_image, flatfield_map, metadata = processor.apply_flatfield_correction(
    raw_image=your_raw_image,
    correction_method='auto',  # Uses position-optimized method
    show_comparison=True
)
```

### Custom Processing Parameters

```python
# Generate flatfield with custom parameters
processor.generate_flatfield_map(
    avg_window=15,        # Larger averaging window
    num_sigma=1.5,        # Custom outlier threshold
    window_length=81,     # Smoothing window length
    polyorder=3           # Polynomial order
)
```

## 🛠️ Installation

### Required Dependencies

```bash
pip install numpy matplotlib scipy pillow pandas
```

### System Requirements

- **Python**: 3.7 or later
- **Memory**: At least 4GB RAM
- **Storage**: Sufficient space for outputs and analysis
- **Display**: GUI backend for matplotlib

## 📊 Technical Details

### Algorithm Overview

1. **Profile Extraction**: Cross-track and along-track brightness profiles at optical center
2. **Sigma Filtering**: Statistical outlier removal with configurable thresholds
3. **Savitzky-Golay Smoothing**: Noise reduction while preserving signal characteristics
4. **Quadratic Envelope Fitting**: Mathematical modeling of illumination patterns
5. **Position-Optimized Correction**: Automatic method selection based on filter characteristics
6. **2D Map Generation**: Additive combination with optical center normalization

### Mathematical Model

Illumination modeled as: `I(x) = ax² + bx + c`

- **a**: Curvature (vignetting effects)
- **b**: Linear gradients (alignment effects)
- **c**: Baseline illumination level

### Quality Metrics

- **Uniformity Improvement**: Percentage reduction in standard deviation
- **Center-to-Edge Ratio**: Spatial uniformity assessment
- **Success Rate**: Percentage of successfully corrected images

## 🔍 Troubleshooting

### Common Issues

**"No images found for filter position"**
- Verify filter position mappings in `Constants.py`
- Check data directory structure and file naming conventions

**"Failed to extract profiles"**
- Adjust `avg_window` parameter (try 5-15)
- Verify optical center coordinates
- Check image signal levels

**"Processing errors"**
- Ensure sufficient memory availability
- Verify write permissions in output directories
- Check all dependencies are installed

### Performance Optimization

**For faster processing:**
- Use smaller averaging windows when possible
- Process positions sequentially
- Reduce smoothing parameters for simple corrections

**For better results:**
- Increase averaging windows for noisy data
- Use higher smoothing parameters for unstable fits
- Verify and adjust optical center coordinates

## 📄 Scientific Documentation

This pipeline generates comprehensive scientific documentation suitable for:

- **Technical reports** with quantitative performance metrics
- **PowerPoint presentations** with publication-ready plots
- **Peer review** with complete methodology documentation
- **Calibration records** with full processing parameter logs

## 🤝 Contributing

This pipeline is designed for scientific reproducibility. When contributing:

1. Maintain comprehensive logging and documentation
2. Preserve metadata in all output files
3. Include robust error handling
4. Update documentation for any functionality changes

## 📞 Contact

**Charlemagne Marc**  
Email: chamrc1@umbc.edu  
AirHARP2 SWIR Calibration Team

## 📝 Version History

- **v2.0.0**: Enhanced position-optimized pipeline with comprehensive validation
- **v1.0.0**: Initial flatfield correction implementation

---

*This tool provides production-ready flatfield correction with 1.5-2% uniformity improvement for the AirHARP2 SWIR instrument. The position-optimized approach ensures maximum correction effectiveness across all filter wavelengths.*