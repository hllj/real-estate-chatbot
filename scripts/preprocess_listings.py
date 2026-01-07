#!/usr/bin/env python3
"""
CLI script for preprocessing real estate listings with amenity-based features.

Usage:
    python scripts/preprocess_listings.py input.csv output.csv
    python scripts/preprocess_listings.py input.csv output.csv --amenities data/hcm_amenities.csv
    python scripts/preprocess_listings.py input.csv output.csv --lat-col lat --lng-col lon
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import AmenityFeatureEngineer


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess real estate listings with amenity-based features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python scripts/preprocess_listings.py data/listings.csv data/listings_enriched.csv

  # Specify amenities file
  python scripts/preprocess_listings.py data/listings.csv output.csv --amenities data/hcm_amenities.csv

  # Custom column names
  python scripts/preprocess_listings.py data/listings.csv output.csv --lat-col lat --lng-col lng

  # Custom radii for amenity counting
  python scripts/preprocess_listings.py data/listings.csv output.csv --radii 200 500 1000 2000
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to input listings CSV file"
    )

    parser.add_argument(
        "output",
        type=str,
        help="Path for output enriched CSV file"
    )

    parser.add_argument(
        "--amenities",
        type=str,
        default="data/hcm_amenities.csv",
        help="Path to amenities CSV file (default: data/hcm_amenities.csv)"
    )

    parser.add_argument(
        "--lat-col",
        type=str,
        default="latitude",
        help="Name of latitude column in listings (default: latitude)"
    )

    parser.add_argument(
        "--lng-col",
        type=str,
        default="longitude",
        help="Name of longitude column in listings (default: longitude)"
    )

    parser.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[300, 500, 1000, 5000],
        help="Radii in meters for amenity counting (default: 300 500 1000 5000)"
    )

    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List all feature names that will be generated and exit"
    )

    parser.add_argument(
        "--list-amenity-types",
        action="store_true",
        help="List all amenity types in the dataset and exit"
    )

    args = parser.parse_args()

    # Initialize feature engineer
    try:
        print(f"Loading amenities from {args.amenities}...")
        engineer = AmenityFeatureEngineer(args.amenities, radii_meters=args.radii)
        print(f"Loaded {len(engineer.amenities)} amenities of {len(engineer.amenity_types)} types")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle info-only modes
    if args.list_amenity_types:
        print("\nAmenity types:")
        for am_type in engineer.get_amenity_types():
            normalized = engineer.type_to_normalized[am_type]
            print(f"  {am_type} -> {normalized}")
        sys.exit(0)

    if args.list_features:
        print("\nFeatures that will be generated:")
        for feature in engineer.get_feature_names():
            print(f"  {feature}")
        sys.exit(0)

    # Process listings
    import pandas as pd

    try:
        print(f"\nLoading listings from {args.input}...")
        listings = pd.read_csv(args.input)
        print(f"Loaded {len(listings)} listings")
    except FileNotFoundError:
        print(f"Error: Listings file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check required columns
    if args.lat_col not in listings.columns:
        print(f"Error: Latitude column '{args.lat_col}' not found in listings", file=sys.stderr)
        print(f"Available columns: {list(listings.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.lng_col not in listings.columns:
        print(f"Error: Longitude column '{args.lng_col}' not found in listings", file=sys.stderr)
        print(f"Available columns: {list(listings.columns)}", file=sys.stderr)
        sys.exit(1)

    # Count valid coordinates
    valid_coords = listings[args.lat_col].notna() & listings[args.lng_col].notna()
    print(f"Listings with valid coordinates: {valid_coords.sum()} / {len(listings)}")

    if not valid_coords.any():
        print("Warning: No listings have valid coordinates. Output will have empty feature columns.")

    # Process
    print("\nComputing amenity features...")
    enriched = engineer.process_listings(listings, lat_col=args.lat_col, lng_col=args.lng_col)

    # Save
    print(f"\nSaving to {args.output}...")
    enriched.to_csv(args.output, index=False)

    print(f"\nDone! Added {len(engineer.get_feature_names())} amenity features.")
    print(f"Output shape: {enriched.shape}")


if __name__ == "__main__":
    main()
