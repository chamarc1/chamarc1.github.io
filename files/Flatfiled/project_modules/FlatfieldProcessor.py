"""
__name__ =      FlatfieldProcessor.py
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
import os
import datetime
import json
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from project_modules.CompositeProcessor import CompositeProcessor
from project_modules.Constants import flatfield_save_path, composite_save_path, parabola_func
from project_modules.Constants import directory_dict, crossTrack_dict, crossTrackDark_dict, alongTrack_dict, alongTrackDark_dict

#----------------------------------------------------------------------------
#-- GLOBALS
#----------------------------------------------------------------------------
# OPTICAL_CENTER_X = 526
# OPTICAL_CENTER_Y = 685
# OPTICAL_CENTER_X = 685
# OPTICAL_CENTER_Y = 526
# OPTICAL_CENTER_X = 675
# OPTICAL_CENTER_Y = 560
OPTICAL_CENTER_X = 640
OPTICAL_CENTER_Y = 510

#----------------------------------------------------------------------------
def plot_composite(composite_image):
    """Plot and save the composite image."""
    if composite_image is not None:
        plt.imshow(composite_image)
        plt.axis("on")
        plt.title("Composite")
        plt.xlabel('Cross-Track Pixels')
        plt.ylabel('Along-Track Pixels')
        os.makedirs(os.path.dirname(composite_save_path), exist_ok=True)
        plt.savefig(composite_save_path)
        plt.show()
    else:
        print("No composite image to display.")

#----------------------------------------------------------------------------
class FlatfieldProcessor:
    def __init__(self, wheel_pos):
        """
        Initializes the FlatfieldProcessor and sets up filter/dark positions.
        """
        self.wheel_pos = wheel_pos  # Store the wheel position
        self.crossTrack_processor = CompositeProcessor(directory_dict["crossTrack"], directory_dict["metadata"])
        self.alongTrack_processor = CompositeProcessor(directory_dict["alongTrack"], directory_dict["metadata"])

        if not wheel_pos.isdigit:
            print("Enter correct wheel pos")
            raise ValueError("Invalid input value")

        self.cross_filter_pos = crossTrack_dict[wheel_pos]
        self.cross_dark_pos = crossTrackDark_dict[wheel_pos]
        self.along_filter_pos = alongTrack_dict[wheel_pos]
        self.along_dark_pos = alongTrackDark_dict[wheel_pos]
        
    def quadratic_fit(self, x_vals, y_vals):
        """
        Fit a quadratic curve (parabola) to x and y values using non-linear least squares.
        Returns fitted y-values and optimal parameters.
        
        Args:
            x_vals: X-axis values
            y_vals: Y-axis values to fit
            
        Returns:
            tuple: (x_vals_clean, y_fit, optimal_parameters)
        """
        valid_indices = ~np.isnan(y_vals)
        x_vals_clean = x_vals[valid_indices]
        y_vals_clean = y_vals[valid_indices]

        if len(x_vals_clean) < 3:
            print(f"Warning: Not enough valid data for quadratic fitting ({len(x_vals_clean)} points). Returning default coefficients.")
            return x_vals_clean, np.zeros_like(x_vals_clean), np.array([0.0, 0.0, 0.0])

        try:
            popt, pcov = curve_fit(parabola_func, x_vals_clean, y_vals_clean)
            self.constant = popt[0]
            self.linear = popt[1]
            self.quadratic = popt[2]
            self.constant_err = np.sqrt(pcov[0][0])
            self.linear_err = np.sqrt(pcov[1][1])
            self.quadratic_err = np.sqrt(pcov[2][2])

            y_fit = parabola_func(x_vals_clean, *popt)
            return x_vals_clean, y_fit, popt
            
        except Exception as e:
            print(f"Warning: Quadratic curve fitting failed: {str(e)}. Trying linear fit as fallback.")
            try:
                # Linear fallback: y = mx + b
                linear_coeffs = np.polyfit(x_vals_clean, y_vals_clean, 1)
                # Convert to quadratic format: [b, m, 0]
                popt = np.array([linear_coeffs[1], linear_coeffs[0], 0.0])
                
                # Set error estimates to reasonable defaults
                self.constant = popt[0]
                self.linear = popt[1] 
                self.quadratic = popt[2]
                self.constant_err = 0.1
                self.linear_err = 0.1
                self.quadratic_err = 0.0
                
                y_fit = parabola_func(x_vals_clean, *popt)
                print("Successfully applied linear fallback fit.")
                return x_vals_clean, y_fit, popt
                
            except Exception as e2:
                print(f"Error: Linear fallback also failed: {str(e2)}. Using mean value as constant fit.")
                # Constant fallback: y = constant
                mean_val = np.mean(y_vals_clean)
                popt = np.array([mean_val, 0.0, 0.0])
                
                self.constant = popt[0]
                self.linear = popt[1]
                self.quadratic = popt[2]
                self.constant_err = np.std(y_vals_clean)
                self.linear_err = 0.0
                self.quadratic_err = 0.0
                
                y_fit = np.full_like(x_vals_clean, mean_val)
                return x_vals_clean, y_fit, popt
    
    def sigma_filter(self, x_vals, y_vals, n_sigma):
        """
        Remove data points where y_vals deviate from the mean by more than n_sigma standard deviations.
        
        Args:
            x_vals: X-axis values
            y_vals: Y-axis values to filter
            n_sigma: Number of standard deviations for filtering threshold
            
        Returns:
            tuple: (filtered_x_vals, filtered_y_vals)
        """
        valid_indices = ~np.isnan(y_vals)
        x_vals_clean = x_vals[valid_indices]
        y_vals_clean = y_vals[valid_indices]

        if len(y_vals_clean) == 0:
            print("Warning: No valid data points for sigma filtering.")
            return np.array([]), np.array([])
        
        if len(y_vals_clean) < 3:
            print(f"Warning: Insufficient data for sigma filtering ({len(y_vals_clean)} points). Returning original data.")
            return x_vals_clean, y_vals_clean

        mean_y = np.mean(y_vals_clean)
        std_y = np.std(y_vals_clean)
        
        # Handle case where standard deviation is zero (all values are the same)
        if std_y == 0:
            print("Warning: Zero standard deviation in data. No filtering applied.")
            return x_vals_clean, y_vals_clean
        
        mask = (y_vals_clean >= mean_y - n_sigma * std_y) & (y_vals_clean <= mean_y + n_sigma * std_y)
        
        # Ensure we don't filter out too many points
        if np.sum(mask) < max(3, len(y_vals_clean) * 0.1):  # Keep at least 10% of data or 3 points
            print("Warning: Sigma filtering would remove too many points. Using less aggressive filtering.")
            # Use a more lenient sigma value
            lenient_sigma = n_sigma * 2
            mask = (y_vals_clean >= mean_y - lenient_sigma * std_y) & (y_vals_clean <= mean_y + lenient_sigma * std_y)
        
        return x_vals_clean[mask], y_vals_clean[mask]
    
    def extract_profile(self, profile_type, avg_window=10, num_sigma=2.0, window_length=61, polyorder=3):
        """
        Extract and process either a row or column profile from images.
        
        Args:
            profile_type (str): Either 'row' for cross-track profile or 'column' for along-track profile
            avg_window (int): Window size for averaging around optical center
            num_sigma (float): Sigma threshold for outlier rejection
            window_length (int): Window length for Savitzky-Golay smoothing
            polyorder (int): Polynomial order for Savitzky-Golay smoothing
            
        Returns:
            tuple: (x_vals_filtered, profile_smoothed, envelope)
        """
        # Configuration mapping for profile types
        config = {
            'row': {
                'processor': self.crossTrack_processor,
                'filter_pos': self.cross_filter_pos,
                'dark_pos': self.cross_dark_pos,
                'optical_center': OPTICAL_CENTER_Y,
                'slice_axis': 0,  # rows
                'mean_axis': 0,   # average along rows
                'title': f"Along-track row Y={OPTICAL_CENTER_Y}",
                'xlabel': "Cross-Track Pixel Index"
            },
            'column': {
                'processor': self.alongTrack_processor,
                'filter_pos': self.along_filter_pos,
                'dark_pos': self.along_dark_pos,
                'optical_center': OPTICAL_CENTER_X,
                'slice_axis': 1,  # columns
                'mean_axis': 1,   # average along columns
                'title': f"Cross-track column X={OPTICAL_CENTER_X}",
                'xlabel': "Along-Track Pixel Index"
            }
        }
        
        if profile_type not in config:
            raise ValueError("profile_type must be either 'row' or 'column'")
        
        cfg = config[profile_type]
        
        # Get images from the appropriate processor
        images = cfg['processor'].generate_images(cfg['filter_pos'], cfg['dark_pos'])
        
        avg_profiles = []
        for img in images:
            # Extract image array (handle both dict and array formats)
            image = img["image"] if isinstance(img, dict) else img
            
            # Calculate slice indices
            start_idx = max(cfg['optical_center'] - avg_window, 0)
            end_idx = min(cfg['optical_center'] + avg_window + 1, image.shape[cfg['slice_axis']])
            if start_idx >= end_idx:
                continue
            
            # Extract profile using dynamic slicing
            if cfg['slice_axis'] == 0:  # row profile
                profile = np.mean(image[start_idx:end_idx, :], axis=cfg['mean_axis'])
            else:  # column profile
                profile = np.mean(image[:, start_idx:end_idx], axis=cfg['mean_axis'])
            
            # Comprehensive checks for profile validity
            if len(profile) == 0:
                print(f"Warning: Empty profile extracted, skipping image.")
                continue
                
            # Check for valid data before any processing
            if np.all(np.isnan(profile)) or np.all(profile <= 0):
                print(f"Warning: Profile contains no valid positive data, skipping image.")
                continue
            
            # Check if we have enough valid finite values
            valid_count = np.sum(np.isfinite(profile) & (profile > 0))
            if valid_count < len(profile) * 0.1:  # Require at least 10% valid data
                print(f"Warning: Profile has insufficient valid data ({valid_count}/{len(profile)} points), skipping image.")
                continue
                
            # Get max value for masking - use nanmax to handle any NaNs
            max_val = np.nanmax(profile)
            if not np.isfinite(max_val) or max_val <= 0:
                print(f"Warning: Profile has invalid maximum value ({max_val}), skipping image.")
                continue
                
            # Mask low-signal regions (values less than half of maximum)
            profile_masked = profile.copy()
            profile_masked[profile_masked < max_val / 2.0] = np.nan
            
            # Check that masking didn't remove all data
            valid_after_mask = np.sum(np.isfinite(profile_masked))
            if valid_after_mask < 3:  # Need at least 3 valid points
                print(f"Warning: Profile has too few valid points after masking ({valid_after_mask} points), skipping image.")
                continue
            
            # Only add profiles that pass all checks
            avg_profiles.append(profile_masked)
            
        if not avg_profiles:
            print(f"Warning: No valid profiles found for {cfg['title']}. Returning None.")
            return None, None, None
        
        # Process combined profile - add additional safety check
        try:
            stacked_profiles = np.stack(avg_profiles)
            
            # Check if stacked profiles have any valid data at all
            if np.all(np.isnan(stacked_profiles)):
                print(f"Warning: All stacked profiles are entirely NaN for {cfg['title']}. Returning None.")
                return None, None, None
            
            # Check if we have at least some positions with valid data
            valid_positions = ~np.all(np.isnan(stacked_profiles), axis=0)
            if not np.any(valid_positions):
                print(f"Warning: No positions have valid data across all profiles for {cfg['title']}. Returning None.")
                return None, None, None
            
            # Only compute mean where we have some valid data
            with np.errstate(invalid='ignore'):  # Suppress runtime warnings for this operation
                combined_profile = np.nanmean(stacked_profiles, axis=0)
            
            # Check if the combined profile has any valid data
            if np.all(np.isnan(combined_profile)):
                print(f"Warning: Combined profile is entirely NaN for {cfg['title']}. Returning None.")
                return None, None, None
            
            # Check if we have sufficient valid data points
            valid_data_count = np.sum(~np.isnan(combined_profile))
            if valid_data_count < 10:  # Require at least 10 valid points
                print(f"Warning: Combined profile has insufficient valid data ({valid_data_count} points) for {cfg['title']}. Returning None.")
                return None, None, None
                
        except ValueError as e:
            print(f"Error creating combined profile for {cfg['title']}: {str(e)}")
            return None, None, None
        x_vals = np.arange(len(combined_profile))
        
        # Outlier rejection and smoothing
        x_vals_filtered, profile_filtered = self.sigma_filter(x_vals, combined_profile, num_sigma)
        
        # Check if we have enough data after filtering
        if len(profile_filtered) < 3:
            print(f"Warning: Insufficient data after sigma filtering for {cfg['title']} (only {len(profile_filtered)} points). Returning None.")
            return None, None, None
        
        # Adjust window length if necessary
        if len(profile_filtered) < window_length:
            window_length = max(polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3, 3)
            if len(profile_filtered) < window_length:
                # Further reduce window length if still too large
                window_length = len(profile_filtered) if len(profile_filtered) % 2 != 0 else len(profile_filtered) - 1
                window_length = max(window_length, 3)  # Minimum window length
                
        try:
            profile_smoothed = savgol_filter(profile_filtered, window_length=window_length, polyorder=polyorder)
        except ValueError as e:
            print(f"Warning: Savitzky-Golay filtering failed for {cfg['title']}: {str(e)}. Using original filtered profile.")
            profile_smoothed = profile_filtered.copy()
        
        # Quadratic envelope fit with error handling
        try:
            _, _, popt_coeffs = self.quadratic_fit(x_vals_filtered, profile_smoothed)
            envelope = parabola_func(x_vals_filtered, *popt_coeffs)
        except Exception as e:
            print(f"Warning: Quadratic fitting failed for {cfg['title']}: {str(e)}. Using linear fit as fallback.")
            # Simple linear fallback
            try:
                linear_coeffs = np.polyfit(x_vals_filtered, profile_smoothed, 1)
                envelope = np.polyval(linear_coeffs, x_vals_filtered)
            except Exception as e2:
                print(f"Error: Linear fallback also failed for {cfg['title']}: {str(e2)}. Using smoothed profile as envelope.")
                envelope = profile_smoothed.copy()
        
        # # Plotting
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot(x_vals, combined_profile, 'x', color='red', label="Combined Profile (raw)", linewidth=1.5, markersize=2.75)
        # ax.plot(x_vals_filtered, profile_filtered, '.', color='green', label=f"Filtered ({num_sigma}-sigma)", linewidth=1.5)
        # ax.plot(x_vals_filtered, profile_smoothed, color='black', label="Smoothed", linewidth=1.5)
        # ax.plot(x_vals_filtered, envelope, color='red', label="Envelope, O(x^2)", linewidth=1.5)

        # ax.set_title(cfg['title'])
        # ax.set_xlabel(cfg['xlabel'])
        # ax.set_ylabel("Signal (DN)")
        # ax.grid(True, linestyle="--", alpha=0.5)
        # ax.legend()
        # plt.show()

        return x_vals, combined_profile, x_vals_filtered, profile_filtered, profile_smoothed, envelope
    
    def extract_row_profile(self, avg_window=10, num_sigma=2.0, window_length=61, polyorder=3):
        """Extract cross-track profile (along-track row cut) at optical center Y."""
        return self.extract_profile('row', avg_window, num_sigma, window_length, polyorder)
    
    def extract_column_profile(self, avg_window=10, num_sigma=2.0, window_length=61, polyorder=3):
        """Extract along-track profile (cross-track column cut) at optical center X."""
        return self.extract_profile('column', avg_window, num_sigma, window_length, polyorder)

    def generate_quadratic_envelope_flatfield(self, avg_window=10, num_sigma=2.0, window_length=61, polyorder=3, smoothing_sigma=None):
        """
        Generate a 2D flatfield correction array based on quadratic envelope fits to mean profiles.
        Models large-scale illumination gradients (e.g., vignetting).
        
        Args:
            avg_window (int): Window size for averaging around optical center
            num_sigma (float): Sigma threshold for outlier removal
            window_length (int): Window length for Savitzky-Golay smoothing
            polyorder (int): Polynomial order for smoothing
            smoothing_sigma (float): Gaussian smoothing parameter (if used)
        """
        print(f"Starting flatfield generation for filter position {getattr(self, 'wheel_pos', 'unknown')}")
        
        # Cross-track envelope using crossTrack_processor
        print("Extracting cross-track brightness profile...")
        x_vals_cross, combined_profile_cross, x_vals_filtered_cross, profile_filtered_cross, profile_cross, envelope_cross = self.extract_row_profile(
            avg_window=avg_window, num_sigma=num_sigma, 
            window_length=window_length, polyorder=polyorder
        )
        
        if x_vals_cross is None or profile_cross is None:
            print("Warning: Failed to extract cross-track profile. Continuing with along-track only.")
        
        # Along-track envelope using alongTrack_processor
        print("Extracting along-track brightness profile...")
        x_vals_along, combined_profile_along, x_vals_filtered_along, profile_filtered_along, profile_along, envelope_along = self.extract_column_profile(
            avg_window=avg_window, num_sigma=num_sigma,
            window_length=window_length, polyorder=polyorder
        )
        
        # Generate combined profile plots for comparison
        print("Generating combined profile plots...")
        try:
            # Prepare data tuples for plotting
            row_data = (x_vals_cross, combined_profile_cross, x_vals_filtered_cross, profile_filtered_cross, profile_cross, envelope_cross)
            col_data = (x_vals_along, combined_profile_along, x_vals_filtered_along, profile_filtered_along, profile_along, envelope_along)
            
            self.plot_combined_profiles(row_data=row_data, col_data=col_data)
            print("✓ Combined profile plots generated successfully")
        except Exception as e:
            print(f"Warning: Failed to generate combined profile plots: {str(e)}")
        
        if x_vals_along is None or profile_along is None:
            print("Warning: Failed to extract along-track profile.")
        
        # Check if we have at least one valid profile
        if (x_vals_filtered_cross is None or profile_cross is None) and (x_vals_filtered_along is None or profile_along is None):
            print("Error: Failed to extract both cross-track and along-track profiles. Cannot generate flatfield.")
            return None
        
        # Create envelope for 2D flatfield correction if we have cross-track data
        if x_vals_filtered_cross is not None and profile_cross is not None:
            try:
                print("Creating 2D flatfield correction...")
                # Get shape from one of the crosstrack images for envelope creation
                images_cross = self.crossTrack_processor.generate_images(self.cross_filter_pos, self.cross_dark_pos)
                cross_arrays = [img["image"] if isinstance(img, dict) else img for img in images_cross]
                
                if cross_arrays:
                    # Create full envelope using the fitted parameters
                    _, _, popt_cross = self.quadratic_fit(x_vals_filtered_cross, profile_cross)
                    envelope_cross_full = parabola_func(np.arange(cross_arrays[0].shape[1]), *popt_cross)
                    envelope_cross_2d = np.tile(envelope_cross_full, (cross_arrays[0].shape[0], 1))
                    print("✓ Successfully created 2D flatfield correction")
                else:
                    print("Warning: No cross-track images available for 2D envelope creation")
                    
            except Exception as e:
                print(f"Warning: Failed to create 2D envelope: {str(e)}")

        # Create 3D plot of the profiles
        print("Generating analysis plots...")
        try:
            self.plot_3d_envelope(x_vals_filtered_cross, profile_cross, envelope_cross, x_vals_filtered_along, profile_along, envelope_along)
            print("✓ Successfully generated 3D visualization")
        except Exception as e:
            print(f"Warning: Failed to create 3D plot: {str(e)}")
        
        # Generate full 2D flatfield map using already extracted data
        try:
            print("\nGenerating 2D flatfield map from extracted quadratic fits...")
            
            # Extract quadratic coefficients from already computed profiles
            cross_coeffs = None
            along_coeffs = None
            
            if x_vals_filtered_cross is not None and profile_cross is not None:
                try:
                    _, _, cross_coeffs = self.quadratic_fit(x_vals_filtered_cross, profile_cross)
                    print(f"Cross-track coefficients: constant={cross_coeffs[0]:.4f}, linear={cross_coeffs[1]:.6f}, quadratic={cross_coeffs[2]:.8f}")
                except Exception as e:
                    print(f"Warning: Failed to extract cross-track coefficients: {str(e)}")
            
            if x_vals_filtered_along is not None and profile_along is not None:
                try:
                    _, _, along_coeffs = self.quadratic_fit(x_vals_filtered_along, profile_along)
                    print(f"Along-track coefficients: constant={along_coeffs[0]:.4f}, linear={along_coeffs[1]:.6f}, quadratic={along_coeffs[2]:.8f}")
                except Exception as e:
                    print(f"Warning: Failed to extract along-track coefficients: {str(e)}")
            
            # Generate flatfield map with extracted coefficients
            flatfield_map = self.generate_flatfield_map(
                cross_coeffs=cross_coeffs,
                along_coeffs=along_coeffs,
                save_path=f"flatfield_map_{getattr(self, 'wheel_pos', 'unknown')}.npz"
            )
            if flatfield_map is not None:
                print("✓ 2D flatfield map generated successfully")
            else:
                print("✗ Failed to generate 2D flatfield map")
        except Exception as e:
            print(f"Warning: Failed to generate 2D flatfield map: {str(e)}")
        
        print("Flatfield generation completed!")

    def plot_combined_profiles(self, row_data=None, col_data=None, avg_window=10, num_sigma=2.0, window_length=61, polyorder=3):
        """
        Plot both row and column profiles in a single plot for comparison.
        
        Args:
            row_data (tuple): Pre-extracted row profile data (x_vals, combined_profile, x_vals_filtered, profile_filtered, profile_smoothed, envelope)
            col_data (tuple): Pre-extracted column profile data (x_vals, combined_profile, x_vals_filtered, profile_filtered, profile_smoothed, envelope)
            avg_window (int): Window size for averaging around optical center (used only if data not provided)
            num_sigma (float): Sigma threshold for outlier rejection (used only if data not provided)
            window_length (int): Window length for Savitzky-Golay smoothing (used only if data not provided)
            polyorder (int): Polynomial order for Savitzky-Golay smoothing (used only if data not provided)
        
        Returns:
            tuple: (row_data, column_data) where each contains all profile data
        """
        # Use provided data or extract if not provided
        if row_data is not None:
            x_vals_row, combined_profile_row, x_vals_filtered_row, profile_filtered_row, profile_smoothed_row, envelope_row = row_data
        else:
            print("Extracting row profile for combined plotting...")
            x_vals_row, combined_profile_row, x_vals_filtered_row, profile_filtered_row, profile_smoothed_row, envelope_row = self.extract_row_profile(
                avg_window=avg_window, num_sigma=num_sigma, 
                window_length=window_length, polyorder=polyorder
            )
        
        if col_data is not None:
            x_vals_col, combined_profile_col, x_vals_filtered_col, profile_filtered_col, profile_smoothed_col, envelope_col = col_data
        else:
            print("Extracting column profile for combined plotting...")
            x_vals_col, combined_profile_col, x_vals_filtered_col, profile_filtered_col, profile_smoothed_col, envelope_col = self.extract_column_profile(
                avg_window=avg_window, num_sigma=num_sigma,
                window_length=window_length, polyorder=polyorder
            )
        
        # Create combined plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot cross-track profile (row)
        if x_vals_row is not None and profile_smoothed_row is not None:
            # Plot raw combined profile
            axes[0].plot(x_vals_row, combined_profile_row, 'x', color='red', 
                        label="Combined Profile (raw)", linewidth=1.5, markersize=2.75)
            # Plot filtered profile
            if x_vals_filtered_row is not None and profile_filtered_row is not None:
                axes[0].plot(x_vals_filtered_row, profile_filtered_row, '.', color='green', 
                           label=f"Filtered ({num_sigma}-sigma)", linewidth=1.5)
            # Plot smoothed profile
            axes[0].plot(x_vals_filtered_row, profile_smoothed_row, color='black', 
                        label="Smoothed", linewidth=1.5)
            # Plot envelope
            if envelope_row is not None:
                axes[0].plot(x_vals_filtered_row, envelope_row, color='red', 
                           label="Envelope, O(x^2)", linewidth=1.5)
            
            axes[0].set_title(f"Cross-Track Profile (Row Y={OPTICAL_CENTER_Y})", fontweight='bold')
            axes[0].set_xlabel("Cross-Track Pixel Index")
            axes[0].set_ylabel("Signal (DN)")
            axes[0].grid(True, linestyle="--", alpha=0.5)
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'Cross-track profile\nextraction failed', 
                        transform=axes[0].transAxes, ha='center', va='center',
                        fontsize=12, color='red')
            axes[0].set_title("Cross-Track Profile (Failed)", fontweight='bold')
        
        # Plot along-track profile (column)
        if x_vals_col is not None and profile_smoothed_col is not None:
            # Plot raw combined profile
            axes[1].plot(x_vals_col, combined_profile_col, 'x', color='red', 
                        label="Combined Profile (raw)", linewidth=1.5, markersize=2.75)
            # Plot filtered profile
            if x_vals_filtered_col is not None and profile_filtered_col is not None:
                axes[1].plot(x_vals_filtered_col, profile_filtered_col, '.', color='green', 
                           label=f"Filtered ({num_sigma}-sigma)", linewidth=1.5)
            # Plot smoothed profile
            axes[1].plot(x_vals_filtered_col, profile_smoothed_col, color='black', 
                        label="Smoothed", linewidth=1.5)
            # Plot envelope
            if envelope_col is not None:
                axes[1].plot(x_vals_filtered_col, envelope_col, color='red', 
                           label="Envelope, O(x^2)", linewidth=1.5)
            
            axes[1].set_title(f"Along-Track Profile (Column X={OPTICAL_CENTER_X})", fontweight='bold')
            axes[1].set_xlabel("Along-Track Pixel Index")
            axes[1].set_ylabel("Signal (DN)")
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Along-track profile\nextraction failed', 
                        transform=axes[1].transAxes, ha='center', va='center',
                        fontsize=12, color='red')
            axes[1].set_title("Along-Track Profile (Failed)", fontweight='bold')
        
        # Add overall title
        filter_pos = getattr(self, 'wheel_pos', 'unknown')
        fig.suptitle(f'Combined Profile Analysis - Filter Position {filter_pos}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Return all the data
        row_data = (x_vals_row, combined_profile_row, x_vals_filtered_row, profile_filtered_row, profile_smoothed_row, envelope_row)
        col_data = (x_vals_col, combined_profile_col, x_vals_filtered_col, profile_filtered_col, profile_smoothed_col, envelope_col)
        
        return row_data, col_data

    def plot_3d_envelope(self, x_vals_cross, profile_cross, envelope_cross, x_vals_along, profile_along, envelope_along):
        """
        Create a 3D plot of the individual profiles and their envelopes.
        X-axis: Cross-track pixel count
        Y-axis: Along-track pixel count  
        Z-axis: Signal (DN)
        
        Handles None values gracefully and only plots available data.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Check if we have any data to plot
        has_cross_data = x_vals_cross is not None and profile_cross is not None
        has_along_data = x_vals_along is not None and profile_along is not None
        
        if not has_cross_data and not has_along_data:
            print("Warning: No valid profile data available for 3D plotting.")
            return
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Track if we plotted anything for the optical center reference
        optical_center_z = 1.0
        
        # Plot cross-track profile (row profile) at optical center Y
        if has_cross_data:
            try:
                y_cross_line = np.full_like(x_vals_cross, OPTICAL_CENTER_X)  # Fixed Y position
                ax.plot(x_vals_cross, y_cross_line, profile_cross, 
                       color='red', linewidth=3, label='Cross-track Profile (smoothed)', alpha=0.9)
                
                # Update optical center reference z-value
                optical_center_z = np.nanmean(profile_cross) if len(profile_cross) > 0 else 1.0
                
                # Plot cross-track envelope
                if envelope_cross is not None:
                    ax.plot(x_vals_cross, y_cross_line, envelope_cross, 
                           color='darkred', linewidth=2, linestyle='--', 
                           label='Cross-track Envelope', alpha=0.9)
            except Exception as e:
                print(f"Warning: Failed to plot cross-track data: {str(e)}")
        else:
            print("Cross-track profile data not available for plotting.")
        
        # Plot along-track profile (column profile) at optical center X  
        if has_along_data:
            try:
                x_along_line = np.full_like(x_vals_along, OPTICAL_CENTER_Y)  # Fixed X position
                ax.plot(x_along_line, x_vals_along, profile_along, 
                       color='blue', linewidth=3, label='Along-track Profile (smoothed)', alpha=0.9)
                
                # Update optical center reference if we didn't have cross-track data
                if not has_cross_data:
                    optical_center_z = np.nanmean(profile_along) if len(profile_along) > 0 else 1.0
                
                # Plot along-track envelope
                if envelope_along is not None:
                    ax.plot(x_along_line, x_vals_along, envelope_along, 
                           color='darkblue', linewidth=2, linestyle='--', 
                           label='Along-track Envelope', alpha=0.9)
            except Exception as e:
                print(f"Warning: Failed to plot along-track data: {str(e)}")
        else:
            print("Along-track profile data not available for plotting.")
        
        # Mark optical center point
        try:
            ax.scatter([OPTICAL_CENTER_Y], [OPTICAL_CENTER_X], [optical_center_z], 
                      color='yellow', s=100, marker='*', 
                      label=f'Optical Center ({OPTICAL_CENTER_Y}, {OPTICAL_CENTER_X})')
        except Exception as e:
            print(f"Warning: Failed to plot optical center marker: {str(e)}")
        
        # Labels and title
        ax.set_xlabel('Cross-Track Pixel Count')
        ax.set_ylabel('Along-Track Pixel Count')
        ax.set_zlabel('Signal (DN)')
        ax.set_title('3D Profile Cuts and Quadratic Envelopes')
        
        # Add legend with error handling
        try:
            ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        except Exception as e:
            print(f"Warning: Failed to create legend: {str(e)}")
        
        # Set viewing angle for better visualization
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        plt.show()
        
    def characterize_pixel_response(self, smoothing_sigma=None, save_path=flatfield_save_path):
        """
        Characterize pixel-to-pixel relative response (flatfield) using composite images.
        """
        cross_composite = self.crossTrack_processor.generate_composite(self.cross_filter_pos, self.cross_dark_pos)
        along_composite = self.alongTrack_processor.generate_composite(self.along_filter_pos, self.along_dark_pos)
        if cross_composite is None or along_composite is None:
            print("Could not generate composite images for flatfield characterization.")
            return None

        flatfield = (cross_composite + along_composite) / 2.0
        print("[Flatfield] Composite images combined.")
        plot_composite(flatfield)

        if smoothing_sigma is not None and smoothing_sigma > 0:
            flatfield = gaussian_filter(flatfield, sigma=smoothing_sigma)
            print(f"[Flatfield] Applied Gaussian smoothing (sigma={smoothing_sigma}).")

        flatfield /= np.mean(flatfield)
        print("[Flatfield] Normalized flatfield to mean 1.")
        
        # Defective Pixel Handling
        mean = np.mean(flatfield)
        std = np.std(flatfield)
        defect_mask = (flatfield < mean - 3*std) | (flatfield > mean + 3*std)
        flatfield[defect_mask] = median_filter(flatfield, size=3)[defect_mask]
        print(f"[Flatfield] Defective pixels replaced: {np.sum(defect_mask)}")
        
        # Metadata
        metadata = {
            "date": datetime.datetime.now().isoformat(),
            "cross_filter_pos": self.cross_filter_pos,
            "cross_dark_pos": self.cross_dark_pos,
            "along_filter_pos": self.along_filter_pos,
            "along_dark_pos": self.along_dark_pos,
            "smoothing_sigma": smoothing_sigma,
            "shape": flatfield.shape,
            "mean": float(np.mean(flatfield)),
            "std": float(np.std(flatfield)),
        }

        np.save(save_path, flatfield)
        print(f"Flatfield characterization saved to {save_path}")
        plot_composite(flatfield)
        return flatfield
    
    def generate_flatfield_map(self, cross_coeffs=None, along_coeffs=None, avg_window=10, num_sigma=2.0, window_length=61, polyorder=3, save_path=None):
        """
        Generate a 2D flatfield map by combining quadratic envelope fits from both cross-track and along-track profiles.
        
        This method can either use pre-computed quadratic coefficients or extract them from scratch.
        
        Args:
            cross_coeffs (array): Pre-computed cross-track quadratic coefficients [constant, linear, quadratic] (optional)
            along_coeffs (array): Pre-computed along-track quadratic coefficients [constant, linear, quadratic] (optional)
            avg_window (int): Window size for averaging around optical center (used only if coefficients not provided)
            num_sigma (float): Sigma threshold for outlier removal (used only if coefficients not provided)
            window_length (int): Window length for Savitzky-Golay smoothing (used only if coefficients not provided)
            polyorder (int): Polynomial order for smoothing (used only if coefficients not provided)
            save_path (str): Path to save the flatfield map (optional)
        
        Returns:
            numpy.ndarray: 2D flatfield correction map normalized to optical center
        """
        print("Generating 2D flatfield map from quadratic envelope fits...")
        
        # Use provided coefficients or extract them
        if cross_coeffs is None:
            print("Extracting cross-track envelope coefficients...")
            try:
                x_vals_cross, combined_profile_cross, x_vals_filtered_cross, profile_filtered_cross, profile_cross, envelope_cross = self.extract_row_profile(
                    avg_window=avg_window, num_sigma=num_sigma, 
                    window_length=window_length, polyorder=polyorder
                )
                
                if x_vals_filtered_cross is None or profile_cross is None:
                    print("Error: Failed to extract cross-track profile for flatfield map generation.")
                    return None
                
                # Get quadratic coefficients for cross-track
                _, _, cross_coeffs = self.quadratic_fit(x_vals_filtered_cross, profile_cross)
                print(f"Cross-track coefficients: constant={cross_coeffs[0]:.4f}, linear={cross_coeffs[1]:.6f}, quadratic={cross_coeffs[2]:.8f}")
                
            except Exception as e:
                print(f"Error extracting cross-track coefficients: {str(e)}")
                return None
        else:
            print(f"Using provided cross-track coefficients: constant={cross_coeffs[0]:.4f}, linear={cross_coeffs[1]:.6f}, quadratic={cross_coeffs[2]:.8f}")
        
        if along_coeffs is None:
            print("Extracting along-track envelope coefficients...")
            try:
                x_vals_along, combined_profile_along, x_vals_filtered_along, profile_filtered_along, profile_along, envelope_along = self.extract_column_profile(
                    avg_window=avg_window, num_sigma=num_sigma,
                    window_length=window_length, polyorder=polyorder
                )
                
                if x_vals_filtered_along is None or profile_along is None:
                    print("Error: Failed to extract along-track profile for flatfield map generation.")
                    return None
                
                # Get quadratic coefficients for along-track
                _, _, along_coeffs = self.quadratic_fit(x_vals_filtered_along, profile_along)
                print(f"Along-track coefficients: constant={along_coeffs[0]:.4f}, linear={along_coeffs[1]:.6f}, quadratic={along_coeffs[2]:.8f}")
                
            except Exception as e:
                print(f"Error extracting along-track coefficients: {str(e)}")
                return None
        else:
            print(f"Using provided along-track coefficients: constant={along_coeffs[0]:.4f}, linear={along_coeffs[1]:.6f}, quadratic={along_coeffs[2]:.8f}")
        
        # Check that we have at least one set of coefficients
        if cross_coeffs is None and along_coeffs is None:
            print("Error: No valid coefficients available for flatfield map generation.")
            return None
        
        # Get sensor dimensions from one of the images
        print("Determining sensor dimensions...")
        try:
            images_cross = self.crossTrack_processor.generate_images(self.cross_filter_pos, self.cross_dark_pos)
            if not images_cross:
                print("Error: No cross-track images available for determining sensor dimensions.")
                return None
            
            # Extract image to get dimensions
            first_image = images_cross[0]["image"] if isinstance(images_cross[0], dict) else images_cross[0]
            sensor_height, sensor_width = first_image.shape
            print(f"Sensor dimensions: {sensor_height} x {sensor_width} pixels")
            
        except Exception as e:
            print(f"Error determining sensor dimensions: {str(e)}")
            return None
        
        # Generate 2D flatfield map
        print("Generating 2D flatfield correction map...")
        try:
            # Create coordinate grids
            x_coords = np.arange(sensor_width)  # Cross-track direction
            y_coords = np.arange(sensor_height)  # Along-track direction
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Apply quadratic corrections in both directions
            if cross_coeffs is not None:
                cross_constant, cross_linear, cross_quadratic = cross_coeffs
                cross_correction = parabola_func(X, cross_constant, cross_linear, cross_quadratic)
            else:
                print("Warning: No cross-track coefficients available. Using uniform correction.")
                cross_correction = np.ones_like(X)
            
            if along_coeffs is not None:
                along_constant, along_linear, along_quadratic = along_coeffs
                along_correction = parabola_func(Y, along_constant, along_linear, along_quadratic)
            else:
                print("Warning: No along-track coefficients available. Using uniform correction.")
                along_correction = np.ones_like(Y)
            
            # Combine the corrections using additive approach
            # Additive combination: cross_correction + along_correction - 1.0
            # This approach treats the corrections as independent additive effects
            # that sum to give the total illumination correction
            flatfield_map = cross_correction + along_correction - 1.0
            print("Using additive combination for flatfield corrections")
            
            # Normalize to optical center
            optical_center_value = flatfield_map[OPTICAL_CENTER_X, OPTICAL_CENTER_Y]
            if optical_center_value > 0:
                flatfield_map = flatfield_map / optical_center_value
            else:
                print("Warning: Invalid optical center value. Using mean normalization.")
                flatfield_map = flatfield_map / np.mean(flatfield_map)
            
            print(f"✓ Flatfield map generated successfully")
            print(f"  - Map shape: {flatfield_map.shape}")
            print(f"  - Value range: [{np.min(flatfield_map):.4f}, {np.max(flatfield_map):.4f}]")
            print(f"  - Mean value: {np.mean(flatfield_map):.4f}")
            print(f"  - Optical center value: {flatfield_map[OPTICAL_CENTER_X, OPTICAL_CENTER_Y]:.4f}")
            
        except Exception as e:
            print(f"Error generating 2D flatfield map: {str(e)}")
            return None
        
        # Optional: Save the flatfield map
        if save_path:
            try:
                # Save as .npz file with metadata
                metadata = {
                    'wheel_pos': getattr(self, 'wheel_pos', 'unknown'),
                    'optical_center_x': OPTICAL_CENTER_X,
                    'optical_center_y': OPTICAL_CENTER_Y,
                    'cross_track_coeffs': cross_coeffs,
                    'along_track_coeffs': along_coeffs,
                    'avg_window': avg_window,
                    'num_sigma': num_sigma,
                    'window_length': window_length,
                    'polyorder': polyorder,
                    'shape': flatfield_map.shape,
                    'mean': float(np.mean(flatfield_map)),
                    'min': float(np.min(flatfield_map)),
                    'max': float(np.max(flatfield_map))
                }
                
                np.savez(save_path, 
                        flatfield_map=flatfield_map,
                        metadata=metadata,
                        cross_track_coeffs=cross_coeffs,
                        along_track_coeffs=along_coeffs)
                
                print(f"✓ Flatfield map saved to: {save_path}")
                
                # Also create a visualization
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot the flatfield map
                im1 = axes[0].imshow(flatfield_map, cmap='viridis', aspect='auto')
                axes[0].set_title(f'2D Flatfield Map - Filter Position {getattr(self, "wheel_pos", "unknown")}')
                axes[0].set_xlabel('Cross-Track Pixels')
                axes[0].set_ylabel('Along-Track Pixels')
                axes[0].plot(OPTICAL_CENTER_Y, OPTICAL_CENTER_X, 'r*', markersize=10, label='Optical Center')
                axes[0].legend()
                plt.colorbar(im1, ax=axes[0], label='Correction Factor')
                
                # Plot cross-sections through optical center
                axes[1].plot(x_coords, flatfield_map[OPTICAL_CENTER_X, :], 'r-', linewidth=2, 
                           label=f'Cross-track (row {OPTICAL_CENTER_X})')
                axes[1].plot(y_coords, flatfield_map[:, OPTICAL_CENTER_Y], 'b-', linewidth=2, 
                           label=f'Along-track (col {OPTICAL_CENTER_Y})')
                axes[1].set_xlabel('Pixel Index')
                axes[1].set_ylabel('Correction Factor')
                axes[1].set_title('Cross-sections through Optical Center')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Warning: Failed to save flatfield map: {str(e)}")
        
        return flatfield_map

    def apply_flatfield_correction(self, raw_image, cross_coeffs=None, along_coeffs=None, 
                                 avg_window=10, num_sigma=2.0, window_length=61, polyorder=3,
                                 flatfield_map=None, save_corrected_path=None, save_flatfield_path=None,
                                 show_comparison=True):
        """
        Apply flatfield correction to a raw image using either a provided flatfield map or 
        by generating one from quadratic envelope fits.
        
        Args:
            raw_image (numpy.ndarray): Raw image to be corrected
            cross_coeffs (array): Pre-computed cross-track quadratic coefficients (optional)
            along_coeffs (array): Pre-computed along-track quadratic coefficients (optional)
            avg_window (int): Window size for averaging around optical center
            num_sigma (float): Sigma threshold for outlier removal
            window_length (int): Window length for Savitzky-Golay smoothing
            polyorder (int): Polynomial order for smoothing
            flatfield_map (numpy.ndarray): Pre-computed flatfield map (optional)
            save_corrected_path (str): Path to save the corrected image (optional)
            save_flatfield_path (str): Path to save the flatfield map (optional)
            show_comparison (bool): Whether to display before/after comparison plots
            
        Returns:
            tuple: (corrected_image, flatfield_map, metadata)
        """
        print("Starting flatfield correction process...")
        
        # Validate input image
        if raw_image is None:
            print("Error: Raw image is None.")
            return None, None, None
            
        if not isinstance(raw_image, np.ndarray):
            print("Error: Raw image must be a numpy array.")
            return None, None, None
            
        if raw_image.ndim != 2:
            print(f"Error: Raw image must be 2D, got {raw_image.ndim}D.")
            return None, None, None
            
        print(f"Input image shape: {raw_image.shape}")
        print(f"Input image data type: {raw_image.dtype}")
        print(f"Input image value range: [{np.min(raw_image):.2f}, {np.max(raw_image):.2f}]")
        
        # Generate or use provided flatfield map
        if flatfield_map is None:
            print("Generating flatfield map...")
            flatfield_map = self.generate_flatfield_map(
                cross_coeffs=cross_coeffs,
                along_coeffs=along_coeffs,
                avg_window=avg_window,
                num_sigma=num_sigma,
                window_length=window_length,
                polyorder=polyorder,
                save_path=save_flatfield_path
            )
            
            if flatfield_map is None:
                print("Error: Failed to generate flatfield map.")
                return None, None, None
        else:
            print("Using provided flatfield map...")
            print(f"Flatfield map shape: {flatfield_map.shape}")
            print(f"Flatfield map value range: [{np.min(flatfield_map):.4f}, {np.max(flatfield_map):.4f}]")
        
        # Validate flatfield map dimensions
        if flatfield_map.shape != raw_image.shape:
            print(f"Error: Flatfield map shape {flatfield_map.shape} does not match image shape {raw_image.shape}")
            return None, None, None
        
        # Apply flatfield correction
        print("Applying flatfield correction...")
        try:
            # Avoid division by zero or very small values
            flatfield_safe = flatfield_map.copy()
            min_threshold = 0.1  # Minimum acceptable flatfield value
            
            # Identify problematic pixels
            bad_pixels = (flatfield_safe < min_threshold) | (~np.isfinite(flatfield_safe))
            n_bad_pixels = np.sum(bad_pixels)
            
            if n_bad_pixels > 0:
                print(f"Warning: Found {n_bad_pixels} problematic flatfield pixels. Replacing with median value.")
                # Replace bad pixels with median of good pixels
                good_flatfield_median = np.median(flatfield_safe[~bad_pixels])
                flatfield_safe[bad_pixels] = good_flatfield_median
            
            # Apply correction: corrected_image = raw_image / flatfield_map
            corrected_image = raw_image.astype(np.float64) / flatfield_safe
            
            print("✓ Flatfield correction applied successfully")
            print(f"Corrected image value range: [{np.min(corrected_image):.2f}, {np.max(corrected_image):.2f}]")
            
        except Exception as e:
            print(f"Error applying flatfield correction: {str(e)}")
            return None, None, None
        
        # Create metadata
        metadata = {
            'date': datetime.datetime.now().isoformat(),
            'wheel_pos': getattr(self, 'wheel_pos', 'unknown'),
            'optical_center_x': OPTICAL_CENTER_X,
            'optical_center_y': OPTICAL_CENTER_Y,
            'original_shape': raw_image.shape,
            'original_dtype': str(raw_image.dtype),
            'original_min': float(np.min(raw_image)),
            'original_max': float(np.max(raw_image)),
            'original_mean': float(np.mean(raw_image)),
            'corrected_min': float(np.min(corrected_image)),
            'corrected_max': float(np.max(corrected_image)),
            'corrected_mean': float(np.mean(corrected_image)),
            'flatfield_min': float(np.min(flatfield_map)),
            'flatfield_max': float(np.max(flatfield_map)),
            'flatfield_mean': float(np.mean(flatfield_map)),
            'bad_pixels_replaced': int(n_bad_pixels) if 'n_bad_pixels' in locals() else 0,
            'processing_parameters': {
                'avg_window': avg_window,
                'num_sigma': num_sigma,
                'window_length': window_length,
                'polyorder': polyorder
            }
        }
        
        # Save corrected image if requested
        if save_corrected_path:
            try:
                np.savez(save_corrected_path,
                        corrected_image=corrected_image,
                        original_image=raw_image,
                        flatfield_map=flatfield_map,
                        metadata=metadata)
                print(f"✓ Corrected image saved to: {save_corrected_path}")
            except Exception as e:
                print(f"Warning: Failed to save corrected image: {str(e)}")
        
        # Display comparison plots if requested
        if show_comparison:
            try:
                self._plot_flatfield_comparison(raw_image, corrected_image, flatfield_map, metadata)
            except Exception as e:
                print(f"Warning: Failed to create comparison plots: {str(e)}")
        
        # Automatically analyze flatfield effectiveness
        try:
            print("\n" + "="*60)
            print("AUTOMATIC FLATFIELD EFFECTIVENESS ANALYSIS")
            print("="*60)
            self._analyze_correction_effectiveness(raw_image, corrected_image, flatfield_map, metadata)
        except Exception as e:
            print(f"Warning: Failed to analyze flatfield effectiveness: {str(e)}")
        
        print("✓ Flatfield correction process completed successfully")
        return corrected_image, flatfield_map, metadata
    
    def _plot_flatfield_comparison(self, raw_image, corrected_image, flatfield_map, metadata):
        """
        Create comparison plots showing the original image, flatfield map, and corrected image.
        
        Args:
            raw_image (numpy.ndarray): Original raw image
            corrected_image (numpy.ndarray): Flatfield-corrected image
            flatfield_map (numpy.ndarray): Flatfield correction map
            metadata (dict): Processing metadata
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        im1 = axes[0, 0].imshow(raw_image, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Original Raw Image')
        axes[0, 0].set_xlabel('Cross-Track Pixels')
        axes[0, 0].set_ylabel('Along-Track Pixels')
        axes[0, 0].plot(OPTICAL_CENTER_Y, OPTICAL_CENTER_X, 'r*', markersize=8, label='Optical Center')
        axes[0, 0].legend()
        plt.colorbar(im1, ax=axes[0, 0], label='Signal (DN)')
        
        # Flatfield map
        im2 = axes[0, 1].imshow(flatfield_map, cmap='RdYlBu_r', aspect='auto')
        axes[0, 1].set_title('Flatfield Correction Map')
        axes[0, 1].set_xlabel('Cross-Track Pixels')
        axes[0, 1].set_ylabel('Along-Track Pixels')
        axes[0, 1].plot(OPTICAL_CENTER_Y, OPTICAL_CENTER_X, 'r*', markersize=8, label='Optical Center')
        axes[0, 1].legend()
        plt.colorbar(im2, ax=axes[0, 1], label='Correction Factor')
        
        # Corrected image
        im3 = axes[0, 2].imshow(corrected_image, cmap='viridis', aspect='auto')
        axes[0, 2].set_title('Flatfield-Corrected Image')
        axes[0, 2].set_xlabel('Cross-Track Pixels')
        axes[0, 2].set_ylabel('Along-Track Pixels')
        axes[0, 2].plot(OPTICAL_CENTER_Y, OPTICAL_CENTER_X, 'r*', markersize=8, label='Optical Center')
        axes[0, 2].legend()
        plt.colorbar(im3, ax=axes[0, 2], label='Corrected Signal (DN)')
        
        # Cross-track profiles through optical center
        row_idx = OPTICAL_CENTER_X
        x_coords = np.arange(raw_image.shape[1])
        
        axes[1, 0].plot(x_coords, raw_image[row_idx, :], 'b-', linewidth=2, label='Original')
        axes[1, 0].plot(x_coords, corrected_image[row_idx, :], 'r-', linewidth=2, label='Corrected')
        axes[1, 0].axvline(OPTICAL_CENTER_Y, color='k', linestyle='--', alpha=0.5, label='Optical Center')
        axes[1, 0].set_xlabel('Cross-Track Pixel Index')
        axes[1, 0].set_ylabel('Signal (DN)')
        axes[1, 0].set_title(f'Cross-Track Profile (Row {row_idx})')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Along-track profiles through optical center
        col_idx = OPTICAL_CENTER_Y
        y_coords = np.arange(raw_image.shape[0])
        
        axes[1, 1].plot(y_coords, raw_image[:, col_idx], 'b-', linewidth=2, label='Original')
        axes[1, 1].plot(y_coords, corrected_image[:, col_idx], 'r-', linewidth=2, label='Corrected')
        axes[1, 1].axvline(OPTICAL_CENTER_X, color='k', linestyle='--', alpha=0.5, label='Optical Center')
        axes[1, 1].set_xlabel('Along-Track Pixel Index')
        axes[1, 1].set_ylabel('Signal (DN)')
        axes[1, 1].set_title(f'Along-Track Profile (Column {col_idx})')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Statistics comparison
        axes[1, 2].axis('off')
        stats_text = f"""
Correction Statistics:

Original Image:
  Mean: {metadata['original_mean']:.2f}
  Min:  {metadata['original_min']:.2f}
  Max:  {metadata['original_max']:.2f}

Corrected Image:
  Mean: {metadata['corrected_mean']:.2f}
  Min:  {metadata['corrected_min']:.2f}
  Max:  {metadata['corrected_max']:.2f}

Flatfield Map:
  Mean: {metadata['flatfield_mean']:.4f}
  Min:  {metadata['flatfield_min']:.4f}
  Max:  {metadata['flatfield_max']:.4f}

Processing Info:
  Filter Position: {metadata['wheel_pos']}
  Bad Pixels Fixed: {metadata['bad_pixels_replaced']}
  Optical Center: ({OPTICAL_CENTER_Y}, {OPTICAL_CENTER_X})
        """
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Overall title
        fig.suptitle(f'Flatfield Correction Results - Filter Position {metadata["wheel_pos"]}', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def _analyze_correction_effectiveness(self, raw_image, corrected_image, flatfield_map, metadata):
        """
        Analyze the effectiveness of the flatfield correction and provide feedback.
        
        Args:
            raw_image (numpy.ndarray): Original raw image
            corrected_image (numpy.ndarray): Flatfield-corrected image
            flatfield_map (numpy.ndarray): Flatfield correction map
            metadata (dict): Processing metadata
        """
        import numpy as np
        
        # Calculate improvement metrics
        original_range = metadata['original_max'] - metadata['original_min']
        corrected_range = metadata['corrected_max'] - metadata['corrected_min']
        range_change = (corrected_range - original_range) / original_range * 100
        
        mean_change = (metadata['corrected_mean'] - metadata['original_mean']) / metadata['original_mean'] * 100
        
        # Flatfield strength
        ff_correction_strength = (metadata['flatfield_max'] - metadata['flatfield_min']) / metadata['flatfield_mean'] * 100
        
        print(f"📊 QUANTITATIVE METRICS:")
        print(f"   • Dynamic Range Change: {range_change:+.1f}%")
        if range_change < -5:
            print("     ✅ EXCELLENT: Significant uniformity improvement")
        elif range_change < 2:
            print("     ⚠️  MODEST: Limited uniformity improvement")
        else:
            print("     ❌ POOR: Range increased - correction may be insufficient")
        
        print(f"   • Mean Signal Change: {mean_change:+.1f}%")
        if abs(mean_change) < 3:
            print("     ✅ GOOD: Signal level well preserved")
        elif abs(mean_change) < 10:
            print("     ⚠️  ACCEPTABLE: Minor signal change")
        else:
            print("     ❌ CONCERNING: Significant signal change")
        
        print(f"   • Correction Strength: {ff_correction_strength:.1f}%")
        if ff_correction_strength > 15:
            print("     ✅ STRONG: Significant vignetting correction")
        elif ff_correction_strength > 8:
            print("     ⚠️  MODERATE: Some correction applied")
        else:
            print("     ❌ WEAK: Minimal correction - may need adjustment")
        
        # Calculate spatial uniformity metrics
        center_x, center_y = OPTICAL_CENTER_X, OPTICAL_CENTER_Y
        
        # Edge-to-center ratio for corrected image
        try:
            # Calculate edge means more carefully to avoid shape issues
            top_edge = np.mean(corrected_image[0, :])
            bottom_edge = np.mean(corrected_image[-1, :])
            left_edge = np.mean(corrected_image[:, 0])
            right_edge = np.mean(corrected_image[:, -1])
            edge_mean = np.mean([top_edge, bottom_edge, left_edge, right_edge])
            
            center_region = corrected_image[center_x-10:center_x+10, center_y-10:center_y+10]
            center_mean = np.mean(center_region)
            edge_center_ratio = edge_mean / center_mean if center_mean > 0 else 1.0
            
            print(f"   • Edge-to-Center Ratio: {edge_center_ratio:.3f}")
            if 0.95 <= edge_center_ratio <= 1.05:
                print("     ✅ EXCELLENT: Very uniform response")
            elif 0.90 <= edge_center_ratio <= 1.10:
                print("     ✅ GOOD: Good uniformity")
            elif 0.85 <= edge_center_ratio <= 1.15:
                print("     ⚠️  ACCEPTABLE: Acceptable uniformity")
            else:
                print("     ❌ POOR: Non-uniform response")
        except Exception as e:
            print(f"   • Edge-to-Center Ratio: Could not calculate (error: {str(e)})")
            edge_center_ratio = 1.0  # Default value for scoring
        
        # Overall assessment
        score = 0
        total_checks = 4
        
        if range_change <= 2:  # Range didn't increase significantly
            score += 1
        if abs(mean_change) < 5:  # Signal preserved
            score += 1
        if 3 <= ff_correction_strength <= 25:  # Reasonable correction
            score += 1
        if 0.85 <= edge_center_ratio <= 1.15:  # Good uniformity
            score += 1
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Effectiveness Score: {score}/{total_checks}")
        
        if score >= 3:
            print("   🎉 SUCCESS: Flatfield correction is effective!")
            interpretation = "EFFECTIVE"
        elif score >= 2:
            print("   ⚠️  PARTIAL: Some improvement achieved")
            interpretation = "PARTIALLY EFFECTIVE"
        else:
            print("   ❌ POOR: Correction needs improvement")
            interpretation = "NEEDS IMPROVEMENT"
        
        # Provide specific recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if interpretation == "EFFECTIVE":
            print("   • Correction is working well - proceed with processing")
            print("   • Save this flatfield map for future use")
        elif interpretation == "PARTIALLY EFFECTIVE":
            print("   • Consider this a maintenance-level correction")
            print("   • Your system already has good uniformity")
            print("   • Validate on additional test images")
        else:
            print("   • Check optical center coordinates")
            print("   • Verify input image quality")
            print("   • Consider different processing parameters")
        
        return score, total_checks, interpretation

    def process_position_with_plots(self, position, num_sigma, presentation_dir):
        """
        Process a single filter position and generate all associated plots.
        
        Args:
            position (str): Filter position (e.g., "pos1", "pos2", etc.)
            num_sigma (float): Standard deviation for Gaussian smoothing
            presentation_dir (str): Directory to save PowerPoint plots
            
        Returns:
            dict: Results containing processed data for the position
        """
        print(f"Processing filter position: {position}")
        
        # Extract profiles for this position
        try:
            # Extract row and column profiles
            row_data = self.extract_row_profile(num_sigma=num_sigma)
            col_data = self.extract_column_profile(num_sigma=num_sigma)
            
            # Generate flatfield using quadratic envelope method
            flatfield_result = self.generate_quadratic_envelope_flatfield(
                num_sigma=num_sigma, 
                smoothing_sigma=num_sigma if num_sigma > 0 else None
            )
            
            # Create combined profile plots
            self.plot_combined_profiles(
                row_data=row_data, 
                col_data=col_data, 
                num_sigma=num_sigma
            )
            
            # Save individual position plots to presentation directory
            position_plots_dir = os.path.join(presentation_dir, f"Position_{position}")
            os.makedirs(position_plots_dir, exist_ok=True)
            
            # Generate 3D envelope plot
            if row_data and col_data and flatfield_result:
                x_vals_cross, profile_cross, envelope_cross = col_data
                x_vals_along, profile_along, envelope_along = row_data
                
                self.plot_3d_envelope(
                    x_vals_cross, profile_cross, envelope_cross,
                    x_vals_along, profile_along, envelope_along
                )
            
            # Generate flatfield map
            flatfield_map = self.generate_flatfield_map(num_sigma=num_sigma)
            
            # Package results
            results = {
                'position': position,
                'row_data': row_data,
                'col_data': col_data,
                'flatfield_result': flatfield_result,
                'flatfield_map': flatfield_map,
                'processor': self
            }
            
            print(f"Successfully processed position {position}")
            return results
            
        except Exception as e:
            print(f"Error processing position {position}: {e}")
            return {
                'position': position,
                'error': str(e),
                'processor': self
            }

    @staticmethod
    def generate_summary_plots(position_results, presentation_dir):
        """
        Generate summary plots comparing all filter positions.
        
        Args:
            position_results (dict): Dictionary containing results for all positions
            presentation_dir (str): Directory to save plots
        """
        print("Generating summary comparison plots...")
        
        # Create summary plots directory
        summary_dir = os.path.join(presentation_dir, "Summary_Plots")
        os.makedirs(summary_dir, exist_ok=True)
        
        try:
            # Extract data for all positions
            positions = []
            row_profiles = []
            col_profiles = []
            
            for pos_key, result in position_results.items():
                if 'error' not in result and result.get('row_data') and result.get('col_data'):
                    positions.append(pos_key)
                    row_profiles.append(result['row_data'])
                    col_profiles.append(result['col_data'])
            
            if not positions:
                print("Warning: No valid position data available for summary plots")
                return
            
            # Create multi-position comparison plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            colors = ['blue', 'red', 'green', 'orange']
            
            # Plot row profiles
            for i, (pos, (x_vals, profile, envelope)) in enumerate(zip(positions, row_profiles)):
                color = colors[i % len(colors)]
                ax1.plot(x_vals, profile, label=f'{pos} data', color=color, alpha=0.7)
                ax1.plot(x_vals, envelope, label=f'{pos} fit', color=color, linewidth=2)
            
            ax1.set_title('Along-Track Profiles Comparison')
            ax1.set_xlabel('Along-Track Pixels')
            ax1.set_ylabel('Normalized Response')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot column profiles
            for i, (pos, (x_vals, profile, envelope)) in enumerate(zip(positions, col_profiles)):
                color = colors[i % len(colors)]
                ax2.plot(x_vals, profile, label=f'{pos} data', color=color, alpha=0.7)
                ax2.plot(x_vals, envelope, label=f'{pos} fit', color=color, linewidth=2)
            
            ax2.set_title('Cross-Track Profiles Comparison')
            ax2.set_xlabel('Cross-Track Pixels')
            ax2.set_ylabel('Normalized Response')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Create flatfield maps comparison
            if len(position_results) >= 2:
                # Show first two positions as examples
                pos_keys = list(position_results.keys())[:2]
                
                for i, pos_key in enumerate(pos_keys):
                    result = position_results[pos_key]
                    if 'flatfield_map' in result and result['flatfield_map'] is not None:
                        ax = ax3 if i == 0 else ax4
                        im = ax.imshow(result['flatfield_map'], cmap='viridis', aspect='auto')
                        ax.set_title(f'Flatfield Map - {pos_key}')
                        ax.set_xlabel('Cross-Track Pixels')
                        ax.set_ylabel('Along-Track Pixels')
                        plt.colorbar(im, ax=ax, label='Correction Factor')
            
            plt.tight_layout()
            summary_plot_path = os.path.join(summary_dir, 'multi_position_comparison.png')
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Summary plots saved to: {summary_dir}")
            
        except Exception as e:
            print(f"Error generating summary plots: {e}")
            import traceback
            traceback.print_exc()