"""
__name__ =      CompositeProcessor.py
__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__credits__ =   ["Charlemagne Marc"]
__version__ =   "1.1.1"
__maintainer__ ="Charlemagne Marc"
__email__ =     "chamrc1@oumbc.edu"
__status__ =    "Production"
"""


#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
import os                             # Provides functions for interacting with the operating system, such as file path manipulation and directory listing.
import numpy as np                    # A fundamental library for numerical computation in Python, essential for working with multi-dimensional arrays representing images.
import matplotlib as mpl              # A comprehensive library for creating static, interactive, and animated visualizations in Python. Used here to set the backend for plotting.
import matplotlib.pyplot as plt       # A module within Matplotlib that provides a MATLAB-like interface for plotting. Used for displaying images and other plots.
from scipy.signal import savgol_filter# A signal processing tool for smoothing data using the Savitzky-Golay filter, which fits successive sub-sets of adjacent data points with a low-degree polynomial by linear least squares.
from scipy.optimize import curve_fit  # Provides functions to use non-linear least squares to fit a function (like a parabola) to data.
from project_modules.ImageProcessor import ImageProcessor # Imports the ImageProcessor class from a local module, likely used for loading and basic processing of individual images.
from project_modules.Constants import composite_save_path, parabola_save_path, flatfield_save_path, flatfield_plot_save_path # Imports predefined file paths for saving generated plots from a local Constants module.
from project_modules.Constants import directory_dict, crossTrack_dict, crossTrackDark_dict, alongTrack_dict, alongTrackDark_dict, parabola_func

#----------------------------------------------------------------------------
#-- GLOBALS
#----------------------------------------------------------------------------
mpl.use("Qt5Agg")                     # Sets the Matplotlib backend to Qt5Agg, which is an interactive backend using the Qt 5 framework. This allows for displaying plots in a separate window.

#----------------------------------------------------------------------------
#-- CompositeProcessor - class for generating composites
#----------------------------------------------------------------------------
class CompositeProcessor:
    def __init__(self, track_dir, metadata):
        """
        Initializes the processor with the directory containing track images and associated metadata.

        :param track_dir: str, the directory containing subdirectories of images (likely organized by filter and degree).
        :param metadata: dict, metadata associated with the image acquisition, such as sensor temperatures during imaging.
        """
        self.track_dir = track_dir
        self.metadata = metadata
        self.track_processor = ImageProcessor(track_dir, metadata)  # Pass metadata to ImageProcessor
        self.constant = 0.0
        self.linear = 0.0
        self.quadratic = 0.0
        self.constant_err = 0.0
        self.linear_err = 0.0
        self.quadratic_err = 0.0

    def compute_average_dark_frame(self, dark_pos):
        """Compute the average dark frame from all images taken at the specified dark position.

        A dark frame is typically an image taken with the sensor's shutter closed, capturing sensor noise and biases. Averaging multiple dark frames reduces random noise.

        :param dark_pos: str, the filter position that corresponds to the dark frame images.
        :return: np.ndarray or None, the average dark frame as a NumPy array. Returns None if no dark images are found.
        """
        dark_images = []
        for degree_pos, images in self.track_processor.image_data.get(dark_pos, {}).items():
            # Extract image arrays from dicts
            dark_images.extend([img_dict["image"] for img_dict in images])
        return np.mean(np.asarray(dark_images), axis=0) if dark_images else None

    def _apply_dark_correction(self, image_array, dark_array):
        """
        Helper method to apply dark frame correction to a single image.
        
        Args:
            image_array: The image to correct (numpy array)
            dark_array: The dark frame to subtract (numpy array)
            
        Returns:
            Corrected image as numpy array
        """
        # Subtract dark frame and take absolute value to avoid negative numbers
        corrected = np.abs(image_array.astype(np.int32) - dark_array.astype(np.int32))
        # Keep values within valid range (0 to 16383 for 14-bit data)
        corrected = np.clip(corrected, 0, 2**14 - 1).astype(np.uint16)
        return corrected

    def correct_images_with_dark_frame(self, images, dark_frame):
        """Correct images by subtracting the average dark frame from each image.

        A dark frame captures sensor noise and should be subtracted from science images
        to improve data quality.

        :param images: list of np.ndarray, a list of images to be corrected.
        :param dark_frame: np.ndarray or None, the average dark frame to subtract.
        :return: list of np.ndarray, a list of corrected images.
        """
        if dark_frame is None:
            return images  # No correction if no dark frame
            
        corrected_images = []
        for image in images:
            corrected_image = self._apply_dark_correction(image, dark_frame)
            corrected_images.append(corrected_image)
        return corrected_images

    def correct_images_pairwise(self, filter_images, dark_images):
        """
        Corrects images by pairing each filter image with a corresponding dark image.
        This method preserves image metadata while applying dark correction.
        """
        n_images = min(len(filter_images), len(dark_images))
        if n_images == 0:
            print("No images to pair for correction.")
            return []
            
        corrected = []
        for i in range(n_images):
            # Extract image arrays and apply correction
            img = filter_images[i]["image"]
            dark = dark_images[i]["image"]
            corrected_img = self._apply_dark_correction(img, dark)
            
            # Preserve metadata from the original filter image
            corrected.append({
                "image": corrected_img,
                "tec": filter_images[i].get("tec"),
                "inttime": filter_images[i].get("inttime"),
                "path": filter_images[i].get("path")
            })
        return corrected

    def correct_images_with_average_dark(self, images, dark_frame):
        """
        Corrects images using an average dark frame while preserving metadata.
        """
        corrected = []
        for img_dict in images:
            img = img_dict["image"]
            corrected_img = self._apply_dark_correction(img, dark_frame)
            
            # Preserve metadata from original image
            corrected.append({
                "image": corrected_img,
                "tec": img_dict.get("tec"),
                "inttime": img_dict.get("inttime"),
                "path": img_dict.get("path")
            })
        return corrected

    def generate_images(self, filter_pos, dark_pos, correction_mode="average"):
        """
        Load and correct images for analysis. 
        
        This method loads images from the specified filter position, applies dark frame
        correction, and returns the corrected images ready for analysis.
        
        Args:
            filter_pos (str): The filter position to load images from
            dark_pos (str): The dark position to use for correction  
            correction_mode (str): Either "average" or "pairwise" correction
            
        Returns:
            list: Corrected images with metadata preserved
        """
        # Check if we have the required image data
        if filter_pos not in self.track_processor.image_data:
            print(f"No images found for filter position: {filter_pos}")
            return []
        if dark_pos not in self.track_processor.image_data:
            print(f"No dark images found for position: {dark_pos}")
            return []

        # Collect all filter images (flatten across different degree positions)
        filter_images = []
        for images in self.track_processor.image_data[filter_pos].values():
            filter_images.extend(images)
        
        # Collect all dark images
        dark_images = []
        for images in self.track_processor.image_data[dark_pos].values():
            dark_images.extend(images)

        # Apply the requested correction method
        if correction_mode == "pairwise":
            corrected = self.correct_images_pairwise(filter_images, dark_images)
            print(f"Applied pairwise correction to {len(corrected)} image pairs.")
        else:  # average mode
            if not dark_images:
                print("No dark images available for average correction.")
                return []
            # Calculate average dark frame
            dark_frame = np.mean(np.stack([img_dict["image"] for img_dict in dark_images]), axis=0)
            corrected = self.correct_images_with_average_dark(filter_images, dark_frame)
            print(f"Applied average dark correction to {len(corrected)} images.")
            
        return corrected

    def generate_composite(self, filter_pos, dark_pos, correction_mode="average"):
        """
        Create a composite (averaged) image from multiple images at the same filter position.
        
        Args:
            filter_pos (str): Filter position to create composite from
            dark_pos (str): Dark position for correction  
            correction_mode (str): Type of dark correction to apply
            
        Returns:
            np.ndarray or None: The composite image, or None if no images found
        """
        # Collect all image arrays for this filter position
        images = []
        for degree_pos, img_dicts in self.track_processor.image_data.get(filter_pos, {}).items():
            for img_dict in img_dicts:
                images.append(img_dict["image"])

        if not images:
            print(f"No images found for filter position: {filter_pos}")
            return None

        # Apply dark correction if using average mode
        if correction_mode == "average":
            dark_frame = self.compute_average_dark_frame(dark_pos)
            if dark_frame is not None:
                images = self.correct_images_with_dark_frame(images, dark_frame)

        # Create composite by averaging all images
        composite = np.mean(np.stack(images), axis=0)
        return composite

    def plotComposite(self, filter_pos, dark_pos):
        """
        Generates and plots the composite image for a specified filter position.

        :param filter_pos: str, the filter position of the desired composite plot.
        :param dark_pos: str, the filter position used for dark frame correction.
        """
        composite_image = self.generate_composite(filter_pos, dark_pos)
        if composite_image is not None: # Checks if a valid composite image was generated.
            plt.imshow(composite_image)
            plt.axis("off")
            plt.show()
        else:
            print("No composite image to display.")

    def find_parabola_core(self, signal):
        """
        Find the central "core" region of a signal where values are above a threshold.

        This is useful for identifying the bright central region of an image profile,
        which often has a parabolic shape in flatfield images.

        Args:
            signal (np.ndarray): 1-D signal to analyze (e.g., a row from an image)
            
        Returns:
            dict: Information about the core region including:
                - peak_value: Maximum value in the signal
                - peak_index: Index of the maximum value  
                - left_index, right_index: Boundaries of core region
                - core_signal: Signal values within the core
                - core_x: X-axis indices for the core region
                - core_width: Width of the core region
        """
        # Find the peak value and its location
        peak_val = np.max(signal)
        peak_idx = np.argmax(signal)
        
        # Set threshold slightly below peak (wider region than half-max)
        threshold = peak_val / 1.15
        
        # Find left boundary of core region
        left_idx = peak_idx
        while left_idx > 0 and signal[left_idx] > threshold:
            left_idx -= 1

        # Find right boundary of core region  
        right_idx = peak_idx
        while right_idx < len(signal) - 1 and signal[right_idx] > threshold:
            right_idx += 1

        # Extract core region information
        core_width = right_idx - left_idx
        core_signal = signal[left_idx:right_idx + 1]
        core_x = np.arange(left_idx, right_idx + 1)

        return {
            "peak_value": peak_val,
            "peak_index": peak_idx,
            "left_index": left_idx,
            "right_index": right_idx,
            "core_signal": core_signal,
            "core_x": core_x,
            "core_width": core_width
        }

    def calculate_FWHM(self, x_vals, y_vals):
        """
        Calculate the Full Width at Half Maximum (FWHM) of a signal.

        FWHM represents the width of the signal where the intensity is at least
        half of the maximum value. This is useful for characterizing the width
        of bright features in images.

        Args:
            x_vals (np.ndarray): X-coordinates (pixel indices)
            y_vals (np.ndarray): Y-values (signal intensities)

        Returns:
            tuple: (x_values, y_values) where the signal is above half maximum
        """
        # Find maximum value and calculate half of it (adjusted threshold)
        peak = np.max(y_vals)
        half_max = peak / 1.15  # Slightly less than half for broader region
        
        # Find all points above the threshold
        above_threshold = y_vals >= half_max
        fwhm_indices = np.where(above_threshold)[0]

        return x_vals[fwhm_indices], y_vals[fwhm_indices]

    def _get_images_from_position(self, position):
        """
        Helper method to collect all images from a given position.
        
        Args:
            position (str): The position to collect images from
            
        Returns:
            list: List of image dictionaries with metadata
        """
        images = []
        for images_in_degree in self.track_processor.image_data.get(position, {}).values():
            images.extend(images_in_degree)
        return images

    def _smooth_signal_if_needed(self, signal, smooth=True, window_length=61, polyorder=3):
        """
        Apply smoothing to a signal if requested and if possible.
        
        Args:
            signal (np.ndarray): Signal to smooth
            smooth (bool): Whether to apply smoothing
            window_length (int): Window size for smoothing filter
            polyorder (int): Polynomial order for smoothing
            
        Returns:
            np.ndarray: Smoothed or original signal
        """
        if not smooth:
            return signal
            
        # Adjust window length if signal is too short
        if window_length >= len(signal):
            window_length = len(signal) - 1 if len(signal) % 2 == 0 else len(signal)
            
        if window_length < polyorder + 1:
            print("Signal too short for smoothing. Returning original signal.")
            return signal
            
        return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

    def plot_parabola_cores(self, filter_pos, dark_pos, along_track_pos, smooth=True, core=True, window_length=61, polyorder=3):
        """
        Plot the core (or full) cross-track signal for each image at a specified along-track row.
        
        This method extracts profiles from images, optionally smooths them, and plots either
        the full profiles or just the core regions.
        """
        images = self.generate_images(filter_pos, dark_pos)
        if not images:
            print("No images to process for plotting.")
            return

        plt.figure(figsize=(14, 6))

        for idx, image_dict in enumerate(images):
            image = image_dict["image"] if isinstance(image_dict, dict) else image_dict
            
            # Extract and average rows around the specified position  
            row_start = max(along_track_pos - 10, 0)
            row_end = min(along_track_pos + 11, image.shape[0])
            
            if row_start >= row_end:
                print(f"Invalid range for image {idx}. Skipping")
                continue

            # Average rows to get cross-track profile
            averaged_row = np.mean(image[row_start:row_end, :], axis=0)
            x_vals = np.arange(len(averaged_row))

            # Apply smoothing if requested
            averaged_row = self._smooth_signal_if_needed(averaged_row, smooth, window_length, polyorder)

            # Plot either core region or full profile
            if core:
                fwhm_x, fwhm_y = self.calculate_FWHM(x_vals, averaged_row)
                plt.plot(fwhm_x, fwhm_y, linewidth=2, label=f"Core Image {idx+1}")
            else:
                plt.plot(x_vals, averaged_row, label=f"Image {idx+1}", linewidth=1.5)

        # Configure plot appearance
        plt.title(f"Cross-Track Profiles (Along-Track Rows {along_track_pos-10} to {along_track_pos+10})")
        plt.xlabel("Cross-Track Pixel Index")
        plt.ylabel("Digital Numbers (DN)")
        plt.grid(True, linestyle="dotted", alpha=0.5)
        plt.ylim(ymin=0)
        plt.legend()
        
        # Save and display plot
        os.makedirs(os.path.dirname(parabola_save_path), exist_ok=True)
        plt.savefig(parabola_save_path)
        plt.show()

    def quadratic_fit(self, x_vals, y_vals):
        """
        Fit a quadratic curve (parabola) to x and y values using non-linear least squares.
        Returns fitted y-values and optimal parameters.
        """
        valid_indices = ~np.isnan(y_vals)
        x_vals_clean = x_vals[valid_indices]
        y_vals_clean = y_vals[valid_indices]

        if len(x_vals_clean) < 3:
            print("Not enough valid data for quadratic fitting. Returning default coefficients.")
            return x_vals_clean, np.zeros_like(x_vals_clean), np.array([0.0, 0.0, 0.0])

        popt, pcov = curve_fit(parabola_func, x_vals_clean, y_vals_clean)
        self.constant = popt[0]
        self.linear = popt[1]
        self.quadratic = popt[2]
        self.constant_err = np.sqrt(pcov[0][0])
        self.linear_err = np.sqrt(pcov[1][1])
        self.quadratic_err = np.sqrt(pcov[2][2])

        y_fit = parabola_func(x_vals_clean, *popt)
        return x_vals_clean, y_fit, popt 

    def sigma_filter(self, x_vals, y_vals, n_sigma):
        """
        Remove data points where y_vals deviate from the mean by more than n_sigma standard deviations.
        """
        valid_indices = ~np.isnan(y_vals)
        x_vals_clean = x_vals[valid_indices]
        y_vals_clean = y_vals[valid_indices]

        if len(y_vals_clean) == 0:
            return np.array([]), np.array([])

        mean_y = np.mean(y_vals_clean)
        std_y = np.std(y_vals_clean, mean=mean_y)
        mask = (y_vals_clean >= mean_y - n_sigma * std_y) & (y_vals_clean <= mean_y + n_sigma * std_y)
        return x_vals_clean[mask], y_vals_clean[mask]

    def get_images_with_metadata_from_path(self, root_path):
        """
        Wrapper for ImageProcessor's recursive image/metadata loader.
        """
        return self.track_processor.get_images_with_metadata_from_path(root_path)
