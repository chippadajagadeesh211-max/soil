"""
SlopeGuard AI — Configuration & Constants
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODULES_DIR = BASE_DIR / "modules"
ASSETS_DIR = BASE_DIR / "assets"

# Dataset file paths
PRIMARY_DATASET_PATH = DATA_DIR / "landslide_modelling_dataset_12409.csv"
HISTORICAL_INVENTORY_PATH = DATA_DIR / "hp_landslide_inventory_clean.csv"
MASTER_FEATURES_PATH = DATA_DIR / "landslide_features_master_6503.csv"
DATA_DICTIONARY_PATH = DATA_DIR / "data_dictionary.csv"
ZIP_DATASET_PATH = DATA_DIR / "A Geo-Environmental Feature Dataset for Landslide.zip"

# Model paths
TRAINED_MODEL_PATH = MODELS_DIR / "trained_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# AI Predictive Feature Columns (Exact features from the 12,409 dataset)
FEATURE_COLUMNS = [
    'Aspect_z', 'Dist_River_z', 'Elevation_z', 'FlowAccumulation_z', 'NDVI_z', 'Rainfall_z', 'Slope_z',
    'aspect_deg', 'elevation_m', 'slope_deg',
    'LULC_10', 'LULC_20', 'LULC_30', 'LULC_40', 'LULC_50', 'LULC_60', 'LULC_70', 'LULC_80', 'LULC_90', 'LULC_100'
]

# Non-predictive columns strictly excluded from training & inference
EXCLUDED_COLUMNS = [
    'sample_id', 'gee_system_index', 'nearest_inventory_sno', 'match_dist_m', 'class'
]

TARGET_COLUMN = 'class'

# Geographic bounds for Himachal Pradesh
HP_CENTER = {"lat": 31.75556, "lon": 77.11000}
HP_BOUNDS = {
    "lat_min": 30.0, "lat_max": 33.5,
    "lon_min": 75.5, "lon_max": 80.0
}

# CSV column alias mappings for user-uploaded CSVs
CSV_ALIASES = {
    'latitude': ['latitude', 'lat', 'latitude_deg', 'lat_deg'],
    'longitude': ['longitude', 'lon', 'long', 'longitude_deg', 'lon_deg'],
    'slope_deg': ['slope_deg', 'slope', 'slope_degrees', 'slope_angle', 'slope_angle_deg'],
    'elevation_m': ['elevation_m', 'elevation', 'altitude', 'elev', 'dem', 'elevation_meters'],
    'aspect_deg': ['aspect_deg', 'aspect', 'aspect_degrees', 'aspect_degree'],
    'Aspect_z': ['aspect_z'],
    'Elevation_z': ['elevation_z', 'elev_z'],
    'Slope_z': ['slope_z'],
    'Rainfall_z': ['rainfall_z', 'rain_z', 'precip_z'],
    'NDVI_z': ['ndvi_z'],
    'Dist_River_z': ['dist_river_z', 'distance_to_river_z', 'drainage_dist_z'],
    'FlowAccumulation_z': ['flowaccumulation_z', 'flow_accumulation_z', 'flow_acc_z'],
    'lulc_code': ['lulc_code', 'landcover_code', 'land_use_code'],
    'city': ['city', 'city_name', 'town', 'municipality']
}

# Geotechnical baseline constants
GAMMA_WATER = 9.81  # kN/m3
