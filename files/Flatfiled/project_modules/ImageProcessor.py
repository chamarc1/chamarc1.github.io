"""
__name__ =      ImageProcessor.py
__author__ =    "Charlemagne Marc"
__copyright__ = "Copyright 2025, ESI SWIR Project"
__credits__ =   ["Charlemagne Marc"]
__version__ =   "1.0.1"
__maintainer__ ="Charlemagne Marc"
__email__ =     "chamrc1@oumbc.edu"
__status__ =    "Production"
"""

#----------------------------------------------------------------------------
#-- IMPORT STATEMENTS
#----------------------------------------------------------------------------
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import glob

#----------------------------------------------------------------------------
#-- GLOBALS
#----------------------------------------------------------------------------
BIT_SHIFT = 2**14
mpl.use("Qt5Agg")

#----------------------------------------------------------------------------
#-- ImageProcessor - class for opening images
#----------------------------------------------------------------------------
class ImageProcessor:
    def __init__(self, base_directory, metadata_csv=None):
        """
        Initializes the ImageProcessor with the base directory and loads all images.
        Optionally loads TEC metadata if a CSV is provided.
        :param base_directory: str, path to the base directory containing image data.
        :param metadata_csv: str, path to the CSV file with metadata.
        """
        self.base_directory = base_directory
        self.tec_map = {}
        if metadata_csv:
            df = pd.read_csv(metadata_csv)
            # Normalize keys for matching
            self.tec_map = {os.path.abspath(str(k)): v for k, v in zip(df['FILEPATH'], df['TEC_READING(CELCIUS)'])}
        self.image_data = self.load_images()

    @staticmethod
    def _extract_integration_time(image_path):
        """
        Extracts the integration time (e.g., '02p0') from the image path.
        Returns None if not found.
        """
        match = re.search(r'INTTIME_([0-9a-zA-Z]+)', image_path)
        return match.group(1) if match else None

    def is_valid_directory(self, path):
        """Returns True if the path is a valid directory."""
        return os.path.isdir(path)

    def process_image(self, image_path):
        """
        Loads an image, converts to NumPy array, and applies bit shift correction.
        Adds TEC and INTTIME metadata.
        """
        try:
            abs_path = os.path.abspath(image_path)
            image = Image.open(abs_path)
            numpy_array = np.asarray(image)
            processed_array = BIT_SHIFT - numpy_array

            # Try absolute path first
            tec_value = self.tec_map.get(abs_path)
            if tec_value is None:
                # Fallback: match by filename only
                abs_filename = os.path.basename(abs_path)
                for k, v in self.tec_map.items():
                    if os.path.basename(k) == abs_filename:
                        tec_value = v
                        break
                if tec_value is None:
                    print(f"TEC lookup failed for: {abs_path}")

            inttime = self._extract_integration_time(abs_path)
            return {
                "image": processed_array,
                "tec": tec_value,
                "inttime": inttime,
                "path": abs_path
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def load_images_from_directory(self, degree_path):
        """
        Loads and processes all TIFF images in a directory.
        :param degree_path: str, path to directory with images for a degree position.
        :return: list of dicts with processed images and metadata
        """
        images = []
        for filename in os.listdir(degree_path):
            if filename.lower().endswith((".tif", ".tiff")):
                image_path = os.path.join(degree_path, filename)
                processed = self.process_image(image_path)
                if processed is not None:
                    images.append(processed)
        return images

    def load_filter_data(self, filter_path):
        """
        Loads image data for a filter position, organized by degree position.
        :param filter_path: str, path to filter directory.
        :return: dict mapping degree position to list of image dicts.
        """
        filter_data = {}
        for degree_pos in os.listdir(filter_path):
            degree_path = os.path.join(filter_path, degree_pos)
            if self.is_valid_directory(degree_path):
                filter_data[degree_pos] = self.load_images_from_directory(degree_path)
        return filter_data

    def load_images(self):
        """
        Loads all images from the base directory, organized by filter and degree position.
        :return: nested dict: filter position -> degree position -> list of image dicts
        """
        image_data = {}
        if not self.is_valid_directory(self.base_directory):
            print(f"Error: Directory not found at {self.base_directory}")
            return image_data

        for filter_pos in os.listdir(self.base_directory):
            filter_path = os.path.join(self.base_directory, filter_pos)
            if self.is_valid_directory(filter_path):
                image_data[filter_pos] = self.load_filter_data(filter_path)
        return image_data

    def show_images(self, filter_pos, degree_pos):
        """
        Displays all images for a given filter and degree position.
        :param filter_pos: str
        :param degree_pos: str
        """
        images = self.get_images(filter_pos, degree_pos)
        if not images:
            print(f"No images found for {filter_pos} at {degree_pos}")
            return

        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]
        for ax, img_dict in zip(axes, images):
            ax.imshow(img_dict["image"])
            ax.axis("off")
        plt.show()

    def get_images(self, filter_pos, degree_pos):
        """
        Returns list of image dicts for a specific filter and degree position.
        :param filter_pos: str
        :param degree_pos: str
        :return: list of np.ndarray
        """
        if filter_pos not in self.image_data or degree_pos not in self.image_data[filter_pos]:
            print(f"No data found for filter position: {filter_pos} or degree position: {degree_pos}")
            return []
        return self.image_data[filter_pos][degree_pos]

    def get_images_with_metadata_from_path(self, root_path):
        """
        Recursively finds all TIFF images under root_path and returns a list of dicts
        with image array, TEC, INTTIME, and path.
        """
        image_dicts = []
        image_paths = glob.glob(os.path.join(root_path, "**", "*.tif*"), recursive=True)
        for image_path in image_paths:
            processed = self.process_image(image_path)
            if processed is not None:
                image_dicts.append(processed)
                # print(processed.get("tec"))
        return image_dicts
