"""
Feature engineering module for real estate amenity-based features.

This module provides functionality to compute location-based features
for property listings based on proximity to various amenities in Ho Chi Minh City.

Features computed:
- Distance to HCM city center
- Minimum distance to each amenity type (markets, schools, hospitals, etc.)
- Count of amenities within various radii (300m, 500m, 1km, 5km)
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# Constants
HCM_CENTER_LAT = 10.776098
HCM_CENTER_LNG = 106.701469
EARTH_RADIUS_KM = 6371.0

# Default radii for amenity counting (in meters)
DEFAULT_RADII_METERS = [300, 500, 1000, 5000]


def haversine_np(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: float, lat2: float) -> np.ndarray:
    """
    Calculate the great circle distance between points using vectorized numpy.

    Args:
        lon1: Array of longitudes for point set 1
        lat1: Array of latitudes for point set 1
        lon2: Single longitude for point 2
        lat2: Single latitude for point 2

    Returns:
        Array of distances in kilometers
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def haversine_single(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points.

    Args:
        lon1: Longitude of point 1
        lat1: Latitude of point 1
        lon2: Longitude of point 2
        lat2: Latitude of point 2

    Returns:
        Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def normalize_name(name: str) -> str:
    """
    Normalize amenity type names to ASCII-safe column names.

    Args:
        name: Original amenity type name (may contain Vietnamese characters)

    Returns:
        Normalized name with underscores, lowercase, ASCII only

    Example:
        >>> normalize_name("Trường Đại học")
        'truong_dai_hoc'
    """
    # Handle Vietnamese-specific characters first
    vietnamese_map = {
        'đ': 'd', 'Đ': 'D',
    }
    result = str(name)
    for viet_char, ascii_char in vietnamese_map.items():
        result = result.replace(viet_char, ascii_char)

    # Normalize unicode characters to remove accents
    nfkd_form = unicodedata.normalize('NFKD', result)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    # Replace spaces with underscores and remove special chars
    clean_name = re.sub(r'[^\w\s]', '', only_ascii).strip().lower()
    return re.sub(r'\s+', '_', clean_name)


class AmenityFeatureEngineer:
    """
    Feature engineering class for computing amenity-based location features.

    This class loads amenity data once and provides methods for both:
    - Batch processing of listings DataFrames
    - Single-point feature computation for real-time inference

    Example:
        >>> engineer = AmenityFeatureEngineer("data/hcm_amenities.csv")
        >>> features = engineer.compute_features_for_point(10.78, 106.70)
        >>> print(features['dist_center_km'])
    """

    def __init__(self, amenities_path: str, radii_meters: Optional[List[int]] = None):
        """
        Initialize the feature engineer with amenities data.

        Args:
            amenities_path: Path to the amenities CSV file
            radii_meters: List of radii (in meters) for counting nearby amenities.
                         Defaults to [300, 500, 1000, 5000]
        """
        self.amenities_path = amenities_path
        self.radii_meters = radii_meters or DEFAULT_RADII_METERS

        # Load and prepare amenities data
        self._load_amenities()
        self._build_trees()

    def _load_amenities(self) -> None:
        """Load and clean amenities data."""
        if not os.path.exists(self.amenities_path):
            raise FileNotFoundError(f"Amenities file not found: {self.amenities_path}")

        self.amenities = pd.read_csv(self.amenities_path)

        # Clean coordinates
        self.amenities['lat'] = pd.to_numeric(self.amenities['lat'], errors='coerce')
        self.amenities['lng'] = pd.to_numeric(self.amenities['lng'], errors='coerce')
        self.amenities = self.amenities.dropna(subset=['lat', 'lng'])

        # Get unique amenity types
        self.amenity_types = self.amenities['type'].unique().tolist()

        # Create normalized type names mapping
        self.type_to_normalized = {
            am_type: normalize_name(am_type) for am_type in self.amenity_types
        }

    def _build_trees(self) -> None:
        """Build BallTree indices for efficient spatial queries."""
        # Convert all amenities to radians for BallTree
        amenities_rad = np.radians(self.amenities[['lat', 'lng']].values)
        self.tree_all = BallTree(amenities_rad, metric='haversine')

        # Build separate trees for each amenity type
        self.trees_by_type: Dict[str, Tuple[BallTree, pd.DataFrame]] = {}

        for am_type in self.amenity_types:
            sub_amenities = self.amenities[self.amenities['type'] == am_type]
            if not sub_amenities.empty:
                sub_rad = np.radians(sub_amenities[['lat', 'lng']].values)
                tree = BallTree(sub_rad, metric='haversine')
                self.trees_by_type[am_type] = (tree, sub_amenities)

    def compute_features_for_point(self, latitude: float, longitude: float) -> Dict[str, float]:
        """
        Compute all amenity-based features for a single point.

        This method is optimized for real-time inference in the chatbot.

        Args:
            latitude: Property latitude
            longitude: Property longitude

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Distance to HCM center
        features['dist_center_km'] = haversine_single(
            longitude, latitude, HCM_CENTER_LNG, HCM_CENTER_LAT
        )

        # Prepare point for BallTree query (needs radians, lat/lng order)
        point_rad = np.radians([[latitude, longitude]])

        # Minimum distance to each amenity type
        for am_type, (tree, _) in self.trees_by_type.items():
            dist_rad, _ = tree.query(point_rad, k=1)
            normalized_type = self.type_to_normalized[am_type]
            features[f'dist_min_{normalized_type}_km'] = dist_rad[0, 0] * EARTH_RADIUS_KM

        # Count amenities within each radius
        for r_m in self.radii_meters:
            r_rad = (r_m / 1000.0) / EARTH_RADIUS_KM
            count = self.tree_all.query_radius(point_rad, r=r_rad, count_only=True)
            features[f'amenities_within_{r_m}m'] = int(count[0])

        return features

    def process_listings(self, listings: pd.DataFrame,
                         lat_col: str = 'latitude',
                         lng_col: str = 'longitude') -> pd.DataFrame:
        """
        Process a DataFrame of listings to add amenity-based features.

        This method is optimized for batch processing of training data.

        Args:
            listings: DataFrame containing property listings
            lat_col: Name of latitude column
            lng_col: Name of longitude column

        Returns:
            DataFrame with added amenity features
        """
        # Make a copy to avoid modifying original
        result = listings.copy()

        # Clean coordinates
        result[lat_col] = pd.to_numeric(result[lat_col], errors='coerce')
        result[lng_col] = pd.to_numeric(result[lng_col], errors='coerce')

        # Track rows with valid coordinates
        valid_mask = result[lat_col].notna() & result[lng_col].notna()

        if not valid_mask.any():
            print("Warning: No valid coordinates found in listings")
            return result

        valid_listings = result[valid_mask]

        # Distance to HCM center
        result.loc[valid_mask, 'dist_center_km'] = haversine_np(
            valid_listings[lng_col].values,
            valid_listings[lat_col].values,
            HCM_CENTER_LNG,
            HCM_CENTER_LAT
        )

        # Prepare coordinates for BallTree queries
        listings_rad = np.radians(valid_listings[[lat_col, lng_col]].values)

        # Minimum distance to each amenity type
        for am_type, (tree, _) in self.trees_by_type.items():
            dist_rad, _ = tree.query(listings_rad, k=1)
            normalized_type = self.type_to_normalized[am_type]
            result.loc[valid_mask, f'dist_min_{normalized_type}_km'] = dist_rad[:, 0] * EARTH_RADIUS_KM

        # Count amenities within each radius
        for r_m in self.radii_meters:
            r_rad = (r_m / 1000.0) / EARTH_RADIUS_KM
            counts = self.tree_all.query_radius(listings_rad, r=r_rad, count_only=True)
            result.loc[valid_mask, f'amenities_within_{r_m}m'] = counts

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be generated.

        Returns:
            List of feature column names
        """
        feature_names = ['dist_center_km']

        # Distance features
        for am_type in self.amenity_types:
            normalized_type = self.type_to_normalized[am_type]
            feature_names.append(f'dist_min_{normalized_type}_km')

        # Count features
        for r_m in self.radii_meters:
            feature_names.append(f'amenities_within_{r_m}m')

        return feature_names

    def get_amenity_types(self) -> List[str]:
        """
        Get list of amenity types available.

        Returns:
            List of amenity type names (original Vietnamese)
        """
        return self.amenity_types.copy()


def process_listings_file(listings_path: str,
                          amenities_path: str,
                          output_path: str,
                          lat_col: str = 'latitude',
                          lng_col: str = 'longitude') -> None:
    """
    Convenience function to process a listings CSV file and save enriched version.

    Args:
        listings_path: Path to input listings CSV
        amenities_path: Path to amenities CSV
        output_path: Path for output enriched CSV
        lat_col: Name of latitude column in listings
        lng_col: Name of longitude column in listings
    """
    print("Loading data...")

    if not os.path.exists(listings_path):
        raise FileNotFoundError(f"Listings file not found: {listings_path}")

    listings = pd.read_csv(listings_path)
    print(f"Listings shape: {listings.shape}")

    # Initialize feature engineer
    engineer = AmenityFeatureEngineer(amenities_path)
    print(f"Loaded {len(engineer.amenities)} amenities of {len(engineer.amenity_types)} types")

    # Process listings
    print("Computing amenity features...")
    enriched = engineer.process_listings(listings, lat_col=lat_col, lng_col=lng_col)

    # Save results
    print(f"Saving to {output_path}...")
    enriched.to_csv(output_path, index=False)

    print(f"Done. Added {len(engineer.get_feature_names())} amenity features.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python feature_engineer.py <listings.csv> <amenities.csv> <output.csv>")
        sys.exit(1)

    process_listings_file(sys.argv[1], sys.argv[2], sys.argv[3])
