# SWIR Image Processing Pipeline for Scientific Data

This project provides a Python-based pipeline for processing scientific SWIR image data, specifically designed for handling multi-frame acquisitions across different experimental conditions. The core functionality includes dark frame correction, flatfield generation and application, composite image creation, flatfiled generation and tools for analyzing cross-track signal profiles.

## Project Structure

The project is organized into the following key Python modules:

* **`Main.py`**: This script is intended as the entry point for running the image processing pipeline. It will import and utilize the classes and functions defined in the other modules to load data, perform processing, and save results.
* **`CompositeProcessor.py`**: Contains the `CompositeProcessor` class, which handles the core image processing tasks. This includes:
    * Loading and managing image data organized by filter and degree.
    * Calculating average dark frames.
    * Correcting images for dark frames.
    * Generating composite images by averaging corrected frames.
    * Generating and applying flatfield corrections.
    * Analyzing and plotting cross-track signal profiles, including core identification and FWHM calculation.
    * Fitting quadratic curves to data.
    * Applying sigma filtering for outlier removal.
* **`ImageProcessor.py`**: Contains the `ImageProcessor` class, responsible for the initial loading and organization of image data from a specified directory structure. It reads image files and stores them in a structured dictionary format based on the directory hierarchy (filter and degree).
* **`Constants.py`**: Defines various constants used throughout the project, such as default file paths for saving generated plots (e.g., composite images, parabola plots, flatfield images).

## Functionality

The project offers the following functionalities:

* **Data Loading and Organization**: Efficiently loads image data from a directory structure, organizing it by experimental parameters like filter position and degree.
* **Dark Frame Correction**: Calculates an average dark frame from specified dark exposures and subtracts it from the science and flatfield images to reduce sensor noise and bias.
* **Flatfield Correction**: Generates a full-image flatfield from dedicated flatfield exposures to correct for pixel sensitivity variations and uneven illumination. This flatfield can then be applied to science images.
* **Composite Image Generation**: Creates high signal-to-noise ratio composite images by averaging multiple dark and flatfield corrected frames acquired under the same experimental conditions.
* **Cross-Track Signal Analysis**: Provides tools to extract and analyze 1D signal profiles (e.g., along a specific row or averaged rows) from the 2D images. This includes:
    * Identifying the "core" region of a signal peak.
    * Calculating the Full Width at Half Maximum (FWHM).
    * Plotting the signal and its core/FWHM.
* **Curve Fitting**: Implements quadratic (parabolic) fitting to analyze trends in the data.
* **Outlier Removal**: Utilizes sigma filtering to identify and remove data points that deviate significantly from the mean.
* **Visualization**: Includes plotting functions to display composite images, cross-track signal profiles, and flatfield images.

## Usage

1.  **Installation**:
    * Ensure you have Python 3.x installed on your system.
    * Install the required Python libraries:
        ```bash
        pip install numpy matplotlib scipy
        ```
    * Place the `CompositeProcessor.py`, `ImageProcessor.py`, and `Constants.py` files in your project directory.

2.  **Data Organization**:
    * Organize your image data in a directory structure that the `ImageProcessor` can understand. Typically, this involves a main directory containing subdirectories for each filter position, and within each filter directory, subdirectories for different degree settings. Ensure your dark frames and flatfield frames are placed in appropriately named filter positions (e.g., "dark", "flat"). To be evetntually updated with the testing files from NASA testing.

3.  **Configuration (`Constants.py`)**:
    * Review and modify the paths defined in `Constants.py` to specify where you want the generated plots to be saved.

4.  **Running the Pipeline (`Main.py`)**:
    * Create a `Main.py` script in your project directory.
    * Import the necessary classes from the project modules:
        ```python
        from CompositeProcessor import CompositeProcessor
        from ImageProcessor import ImageProcessor
        from Constants import *
        import os
        ```
    * Instantiate the `CompositeProcessor`, providing the path to your main image directory and any relevant metadata:
        ```python
        track_directory = '/path/to/your/image/data'
        metadata = {} # Add any relevant metadata here
        processor = CompositeProcessor(track_directory, metadata)
        ```
    * Use the methods of the `CompositeProcessor` to perform the desired processing steps:
        ```python
        # Generate a flatfield image
        flatfield_image = processor.generate_flatfield("flat", "dark")
        if flatfield_image is not None:
            processor.plot_flatfield(flatfield_image)

        # Generate a composite image with flatfield correction
        composite = processor.generate_composite("science_filter_1", "dark", flatfield=flatfield_image)
        if composite is not None:
            plot_composite(composite) # Assuming plot_composite is in this script or imported

        # Analyze cross-track profiles
        processor.plot_parabola_cores("science_filter_1", "dark", along_track_pos=50)
        ```
    * Customize `Main.py` according to your specific data and analysis requirements.

## Contributing

Contact Charlemagne Marc (chamarc1@umbc.edu) for any contriubtion inquiries.