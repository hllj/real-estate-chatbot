"""
Real Estate Price Predictor using trained XGBoost model.

This module provides a unified interface for predicting property prices
using the trained ML model, integrating with PropertyFeatures from the chatbot
and computing amenity-based features from coordinates.
"""

import logging
import math
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Set up logger for this module
logger = logging.getLogger(__name__)

from src.models import PropertyFeatures
from src.preprocessing.feature_engineer import AmenityFeatureEngineer
from src.ml.explainer import ShapExplainer, create_explainer


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
# Model paths for Sell and Rent modes
MODEL_PATH_SELL = PROJECT_ROOT / "models" / "s_with_features_best_model.pkl"
MODEL_PATH_RENT = PROJECT_ROOT / "models" / "u_with_features_best_model.pkl"
MODEL_PATH = MODEL_PATH_SELL  # Default for backward compatibility
AMENITIES_PATH = PROJECT_ROOT / "data" / "hcm_amenities.csv"
RMSE_LOG = 0.1340  # Typical RMSE in log10 price from model training

# Default values for missing features (computed from training data statistics)
# These are median/mode values from the training dataset
DEFAULT_VALUES = {
    # Numeric defaults (median values from training data)
    "floornumber": None,  # Only ~3% have this, leave as None for imputer
    "floors": None,  # ~46% have this, leave as None for imputer
    "length": None,  # ~71% have this
    "living_size": None,  # ~37% have this
    "width": None,  # ~73% have this
    "log_size": 1.7,  # Median log10 of ~50m² = 1.7

    # Rent-specific numeric defaults
    "is_good_room": None,  # 0 or 1, leave as None for imputer
    "deposit": None,  # Deposit amount, leave as None for imputer

    # Categorical defaults (mode values)
    "is_main_street": "nan",  # Missing value marker used in training data
    "apartment_type_name": "Không có thông tin",
    "property_legal_document_status": "Không có thông tin",
    "rooms_count": "Không có thông tin",
    "toilets_count": "Không có thông tin",
    "furnishing_sell_status": "Không có thông tin",
    "furnishing_rent_status": "Không có thông tin",
    "balconydirection_name": "Không có thông tin",
    "direction_name": "Không có thông tin",
    "house_type_name": "Không có thông tin",
    "commercial_type_name": "Không có thông tin",
    "land_type_name": "Không có thông tin",
    "property_status_name": "Không có thông tin",
}

# District centroid coordinates (fallback when geocoding unavailable)
# These are approximate centroids for each district in Ho Chi Minh City
DISTRICT_CENTROIDS = {
    "Quận 1": (10.7756, 106.7019),
    "Quận 3": (10.7833, 106.6833),
    "Quận 4": (10.7578, 106.7014),
    "Quận 5": (10.7544, 106.6628),
    "Quận 6": (10.7478, 106.6353),
    "Quận 7": (10.7340, 106.7218),
    "Quận 8": (10.7240, 106.6284),
    "Quận 10": (10.7731, 106.6678),
    "Quận 11": (10.7652, 106.6502),
    "Quận 12": (10.8671, 106.6413),
    "Thành phố Thủ Đức": (10.8700, 106.8030),
    "Quận Tân Phú": (10.7918, 106.6280),
    "Huyện Củ Chi": (10.9738, 106.4932),
    "Quận Bình Tân": (10.7652, 106.6036),
    "Huyện Bình Chánh": (10.6833, 106.5833),
    "Huyện Hóc Môn": (10.8867, 106.5900),
    "Quận Gò Vấp": (10.8386, 106.6652),
    "Quận Bình Thạnh": (10.8105, 106.7091),
    "Huyện Cần Giờ": (10.4114, 106.9536),
    "Quận Tân Bình": (10.8014, 106.6528),
    "Huyện Nhà Bè": (10.6833, 106.7333),
    "Quận Phú Nhuận": (10.7994, 106.6819),
}


class RealEstatePricePredictor:
    """
    Price predictor that integrates the trained XGBoost/Random Forest model with
    the chatbot's PropertyFeatures and amenity feature engineering.
    Supports both Sell and Rent modes.
    """

    # Expected feature columns (in order expected by the model)
    NUMERIC_FEATURES_BASE = [
        "floornumber", "floors", "length", "living_size", "width",
        "longitude", "latitude", "log_size",
        "dist_center_km",
        "dist_min_cho_km", "dist_min_sieu_thi_km", "dist_min_sieu_thi_mini_km",
        "dist_min_cua_hang_tien_loi_km", "dist_min_truong_mam_non_km",
        "dist_min_truong_tieu_hoc_km", "dist_min_truong_thpt_km",
        "dist_min_truong_ai_hoc_km", "dist_min_benh_vien_km",
        "dist_min_tram_y_te_km", "dist_min_cong_vien_km",
        "dist_min_san_van_ong_km", "dist_min_khu_vui_choi_km",
        "dist_min_pho_i_bo_km", "dist_min_quan_an_km",
        "dist_min_nha_hang_km", "dist_min_quan_cafe_km",
        "dist_min_trung_tam_thuong_mai_km", "dist_min_ngan_hang_km",
        "dist_min_atm_km", "dist_min_metro_km",
        "dist_min_ben_xe_km", "dist_min_ga_tau_km",
        "amenities_within_300m", "amenities_within_500m",
        "amenities_within_1000m", "amenities_within_5000m",
    ]

    # Rent-specific numeric features
    RENT_SPECIFIC_FEATURES = ["is_good_room", "deposit"]

    # For backward compatibility
    NUMERIC_FEATURES = NUMERIC_FEATURES_BASE

    # Base categorical features (without furnishing - added dynamically based on mode)
    CATEGORICAL_FEATURES_BASE = [
        "area_name", "category_name", "is_main_street",
        "apartment_type_name", "property_legal_document_status",
        "rooms_count", "toilets_count",
        "balconydirection_name", "direction_name",
        "house_type_name", "commercial_type_name",
        "land_type_name", "property_status_name",
    ]

    # Furnishing field for each mode
    FURNISHING_FIELD_SELL = "furnishing_sell_status"
    FURNISHING_FIELD_RENT = "furnishing_rent_status"

    def __init__(self, model_path: str = None, amenities_path: str = None, mode: str = "Sell"):
        """
        Initialize the predictor with model and amenity data.

        Args:
            model_path: Path to saved model. If not specified, uses mode-appropriate default.
            amenities_path: Path to amenities CSV. Defaults to data/hcm_amenities.csv
            mode: "Sell" or "Rent" - determines which model and furnishing field to use
        """
        self.mode = mode

        # Determine model path based on mode if not explicitly provided
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = MODEL_PATH_SELL if mode == "Sell" else MODEL_PATH_RENT

        self.amenities_path = Path(amenities_path) if amenities_path else AMENITIES_PATH

        self._model = None
        self._amenity_engineer = None
        self._shap_explainer = None

    @property
    def furnishing_field(self) -> str:
        """Get the furnishing field name based on mode."""
        return self.FURNISHING_FIELD_SELL if self.mode == "Sell" else self.FURNISHING_FIELD_RENT

    @property
    def numeric_features(self) -> List[str]:
        """Get numeric features list based on mode. Rent mode includes additional fields."""
        if self.mode == "Rent":
            return self.NUMERIC_FEATURES_BASE + self.RENT_SPECIFIC_FEATURES
        return self.NUMERIC_FEATURES_BASE

    @property
    def categorical_features(self) -> List[str]:
        """Get categorical features list with correct furnishing field based on mode."""
        # Insert furnishing field after toilets_count (index 7 in base list)
        features = self.CATEGORICAL_FEATURES_BASE.copy()
        # Insert after toilets_count which is at index 6
        features.insert(7, self.furnishing_field)
        return features

    # Keep CATEGORICAL_FEATURES as property for backward compatibility
    @property
    def CATEGORICAL_FEATURES(self) -> List[str]:
        """Backward compatible property that returns mode-appropriate categorical features."""
        return self.categorical_features

    @property
    def model(self):
        """Lazy load the model on first access."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            print(f"Loading trained model from {self.model_path}")
            self._model = joblib.load(self.model_path)
            print("Trained model loaded successfully")
        return self._model

    @property
    def amenity_engineer(self) -> AmenityFeatureEngineer:
        """Lazy load the amenity feature engineer on first access."""
        if self._amenity_engineer is None:
            if not self.amenities_path.exists():
                raise FileNotFoundError(f"Amenities file not found: {self.amenities_path}")
            print(f"Loading amenity data from {self.amenities_path}")
            self._amenity_engineer = AmenityFeatureEngineer(str(self.amenities_path))
            print("Amenity feature engineer loaded successfully")
        return self._amenity_engineer

    @property
    def shap_explainer(self) -> Optional[ShapExplainer]:
        """Lazy load the SHAP explainer on first access."""
        if self._shap_explainer is None:
            try:
                # Get all feature names for the explainer
                all_features = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
                print("Initializing SHAP explainer...")
                self._shap_explainer = create_explainer(self.model, all_features)
                if self._shap_explainer:
                    print("SHAP explainer initialized successfully")
                else:
                    print("SHAP explainer not available")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {e}")
                self._shap_explainer = None
        return self._shap_explainer

    def _compute_size_and_log_size(self, features: PropertyFeatures) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute size and log_size from available property features.

        Returns:
            Tuple of (size, log_size)
        """
        size = None

        # Priority: size > living_size > width*length
        if features.size is not None:
            size = features.size
        elif features.living_size is not None:
            size = features.living_size
        elif features.width is not None and features.length is not None:
            size = features.width * features.length

        if size is not None and size > 0:
            log_size = math.log10(size)
        else:
            log_size = DEFAULT_VALUES.get("log_size", 1.7)

        return size, log_size

    def _get_coordinates(self, features: PropertyFeatures) -> Tuple[Optional[float], Optional[float]]:
        """
        Get coordinates from features or fallback to district centroid.

        Returns:
            Tuple of (latitude, longitude)
        """
        # Use provided coordinates if available
        if features.latitude is not None and features.longitude is not None:
            return features.latitude, features.longitude

        # Fallback to district centroid
        if features.area_name and features.area_name in DISTRICT_CENTROIDS:
            return DISTRICT_CENTROIDS[features.area_name]

        # Default to HCM city center
        return 10.7756, 106.7019

    def _features_to_dataframe(self, features: PropertyFeatures) -> pd.DataFrame:
        """
        Convert PropertyFeatures to a DataFrame with all 50 input features.

        The model expects 50 features (51 total columns including log_price target).
        """
        # Start with empty row
        data = {}

        # Get coordinates (required for amenity features)
        latitude, longitude = self._get_coordinates(features)

        # Compute amenity features from coordinates
        amenity_features = self.amenity_engineer.compute_features_for_point(latitude, longitude)

        # Compute log_size
        _, log_size = self._compute_size_and_log_size(features)

        # --- Numeric Features ---
        data["floornumber"] = features.floornumber
        data["floors"] = features.floors
        data["length"] = features.length
        data["living_size"] = features.living_size
        data["width"] = features.width
        data["longitude"] = longitude
        data["latitude"] = latitude
        data["log_size"] = log_size

        # Add all amenity distance and count features
        for feature_name in self.NUMERIC_FEATURES_BASE:
            if feature_name.startswith("dist_") or feature_name.startswith("amenities_"):
                data[feature_name] = amenity_features.get(feature_name)

        # --- Categorical Features ---
        data["area_name"] = features.area_name
        data["category_name"] = features.category_name or DEFAULT_VALUES["category_name"] if hasattr(DEFAULT_VALUES, "category_name") else None

        # Handle is_main_street (convert bool to string expected by model: "True"/"False"/"nan")
        if features.is_main_street is not None:
            data["is_main_street"] = "True" if features.is_main_street else "False"
        else:
            data["is_main_street"] = "nan"

        data["apartment_type_name"] = features.apartment_type_name or DEFAULT_VALUES.get("apartment_type_name")
        data["property_legal_document_status"] = features.property_legal_document_status or DEFAULT_VALUES.get("property_legal_document_status")
        data["rooms_count"] = features.rooms_count or DEFAULT_VALUES.get("rooms_count")
        data["toilets_count"] = features.toilets_count or DEFAULT_VALUES.get("toilets_count")

        # Use correct furnishing field based on mode
        if self.mode == "Sell":
            data["furnishing_sell_status"] = features.furnishing_sell_status or DEFAULT_VALUES.get("furnishing_sell_status")
        else:  # Rent mode
            data["furnishing_rent_status"] = features.furnishing_rent_status or DEFAULT_VALUES.get("furnishing_rent_status")
            # Add rent-specific numeric features
            data["is_good_room"] = features.is_good_room if features.is_good_room is not None else DEFAULT_VALUES.get("is_good_room")
            data["deposit"] = features.deposit if features.deposit is not None else DEFAULT_VALUES.get("deposit")

        data["balconydirection_name"] = features.balconydirection_name or DEFAULT_VALUES.get("balconydirection_name")
        data["direction_name"] = features.direction_name or DEFAULT_VALUES.get("direction_name")
        data["house_type_name"] = features.house_type_name or DEFAULT_VALUES.get("house_type_name")
        data["commercial_type_name"] = features.commercial_type_name or DEFAULT_VALUES.get("commercial_type_name")
        data["land_type_name"] = features.land_type_name or DEFAULT_VALUES.get("land_type_name")
        data["property_status_name"] = features.property_status_name or DEFAULT_VALUES.get("property_status_name")

        # Create DataFrame with proper column order (use properties for mode-specific features)
        all_features = self.numeric_features + self.categorical_features
        df = pd.DataFrame([data])

        # Ensure all columns exist and are in correct order
        for col in all_features:
            if col not in df.columns:
                df[col] = None

        return df[all_features]

    def predict(self, features: PropertyFeatures) -> Optional[float]:
        """
        Predict property price from PropertyFeatures.

        Args:
            features: PropertyFeatures object from chatbot

        Returns:
            Predicted price in VND, or None if prediction fails
        """
        try:
            print(f"Making prediction with trained model for area: {features.area_name}")

            # Convert to DataFrame
            df = self._features_to_dataframe(features)
            print(f"Feature DataFrame shape: {df.shape}")
            print(f"Feature DataFrame preview:\n{df.head()}")

            # Make prediction (model outputs log_price)
            log_price = self.model.predict(df)[0]
            print(f"Model predicted log_price: {log_price}")

            # Convert from log10 to actual price
            price = 10 ** log_price
            print(f"Trained model prediction: {price:,.0f} VND")

            return float(price)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def predict_with_confidence(self, features: PropertyFeatures) -> Dict[str, Any]:
        """
        Predict property price with confidence score and SHAP explanation.

        Args:
            features: PropertyFeatures object from chatbot

        Returns:
            Dictionary with prediction, confidence score, and SHAP explanation
        """
        try:
            print(f"Making prediction with confidence for area: {features.area_name}")
            df = self._features_to_dataframe(features)
            log_price = self.model.predict(df)[0]
            price = 10 ** log_price

            # Get the pipeline and model
            pipeline = self.model
            model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline

            # Compute confidence score based on prediction consistency
            confidence = 0.75  # Default confidence score (75%)

            if hasattr(model, 'estimators_'):
                # Ensemble model (Random Forest, Gradient Boosting, etc.)
                # Compute confidence from consistency of tree predictions
                tree_predictions = []
                preprocessed_X = pipeline.named_steps['preprocessor'].transform(df)

                for estimator in model.estimators_:
                    pred = estimator.predict(preprocessed_X)[0]
                    tree_predictions.append(pred)  # Keep in log scale for std calculation

                # Calculate coefficient of variation (CV) in log space
                std_log = float(np.std(tree_predictions))
                # Convert to confidence: lower std = higher confidence
                # Using exponential decay: confidence = exp(-k * std)
                # Calibrated so that std=0 -> 95%, std=0.2 -> ~70%, std=0.4 -> ~50%
                confidence = float(0.95 * np.exp(-3.5 * std_log))
                confidence = max(0.30, min(0.95, confidence))  # Clamp between 30% and 95%
                print(f"Prediction with confidence (from {len(model.estimators_)} estimators): {price:,.0f} VND (confidence: {confidence:.1%})")
            else:
                # Fallback: Use RMSE-based confidence estimation
                # Lower RMSE = higher confidence
                rmse_log = RMSE_LOG
                confidence = float(0.90 * np.exp(-2.5 * rmse_log))
                confidence = max(0.30, min(0.90, confidence))
                print(f"Prediction with confidence (RMSE-based): {price:,.0f} VND (confidence: {confidence:.1%})")

            # Compute SHAP explanation
            shap_explanation = None
            if self.shap_explainer is not None:
                try:
                    print("Computing SHAP explanation...")
                    shap_explanation = self.shap_explainer.explain_prediction(df, top_n=30)  # Get all features
                    if shap_explanation.get('success'):
                        print(f"SHAP explanation computed with {len(shap_explanation.get('top_features', []))} top features")
                    else:
                        print(f"SHAP explanation failed: {shap_explanation.get('error')}")
                        shap_explanation = None
                except Exception as e:
                    logger.error(f"SHAP explanation error: {e}")
                    shap_explanation = None

            return {
                "predicted_price": float(price),
                "log_price": float(log_price),
                "confidence": confidence,
                "features_used": {
                    "has_coordinates": features.latitude is not None and features.longitude is not None,
                    "has_size": features.size is not None or features.living_size is not None or (features.width and features.length),
                    "area_name": features.area_name,
                },
                "shap_explanation": shap_explanation,
            }

        except Exception as e:
            logger.error(f"Prediction with confidence failed: {e}")
            return {
                "predicted_price": None,
                "error": str(e),
            }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model (if available).

        Returns:
            DataFrame with feature names and importance scores
        """
        try:
            print("Getting feature importance from model")
            model = self.model

            # Get the actual model from pipeline
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                estimator = model.named_steps["model"]
            else:
                estimator = model

            if hasattr(estimator, "feature_importances_"):
                # Get feature names from preprocessor
                all_features = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES

                # Note: After one-hot encoding, there will be more features
                # This is a simplified version showing original feature groups
                return pd.DataFrame({
                    "feature": all_features,
                    "importance": estimator.feature_importances_[:len(all_features)]
                }).sort_values("importance", ascending=False)

            print("Model does not have feature_importances_ attribute")
            return None

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None


# Singleton instances for reuse (lazy loaded) - one per mode
_predictor_instances: Dict[str, RealEstatePricePredictor] = {}


def get_predictor(mode: str = "Sell") -> RealEstatePricePredictor:
    """Get or create the singleton predictor instance for the given mode."""
    global _predictor_instances
    if mode not in _predictor_instances:
        _predictor_instances[mode] = RealEstatePricePredictor(mode=mode)
    return _predictor_instances[mode]


class PricePredictor:
    """
    Backward-compatible wrapper that matches the placeholder_model interface.

    This class provides the same interface as the original PricePredictor
    to minimize changes in the graph nodes. Includes fallback to heuristic
    model if the trained model can't be loaded.
    Supports both Sell and Rent modes.
    """

    def __init__(self, mode: str = "Sell"):
        self.mode = mode
        self._predictor = None
        self._use_fallback = False

        try:
            print(f"Initializing PricePredictor in {mode} mode, attempting to load trained model...")
            self._predictor = get_predictor(mode)
            # Test if model can be loaded
            _ = self._predictor.model
            print(f"PricePredictor initialized with trained model for {mode} mode")
        except Exception as e:
            print(f"Could not load trained model for {mode} mode ({e}). Using fallback heuristic model.")
            self._use_fallback = True

    def _fallback_predict(self, features: PropertyFeatures) -> float:
        """
        Fallback heuristic prediction when trained model is unavailable.
        Uses simple price per m² estimation based on location and mode.
        """
        print(f"Using FALLBACK heuristic model for area: {features.area_name} (mode: {self.mode})")

        # Base price differs significantly between Sell and Rent modes
        if self.mode == "Sell":
            base_price_per_m2 = 50_000_000  # 50 million VND/m² for sale
        else:  # Rent mode - monthly rent
            base_price_per_m2 = 200_000  # 200k VND/m²/month for rent

        # Get size
        size = features.size or features.living_size
        if size is None and features.width and features.length:
            size = features.width * features.length
        if size is None:
            size = 50  # Default size

        # Location factors (based on HCM City market averages)
        location_factors = {
            "Quận 1": 3.5,
            "Quận 3": 2.5,
            "Quận 4": 1.8,
            "Quận 5": 2.0,
            "Quận 7": 2.2,
            "Quận Bình Thạnh": 1.8,
            "Quận Phú Nhuận": 2.0,
            "Thành phố Thủ Đức": 1.5,
            "Quận Gò Vấp": 1.4,
            "Quận Tân Bình": 1.6,
            "Quận Tân Phú": 1.3,
            "Quận 10": 1.8,
            "Quận 11": 1.5,
            "Quận 12": 1.2,
            "Quận 6": 1.4,
            "Quận 8": 1.3,
            "Quận Bình Tân": 1.2,
            "Huyện Bình Chánh": 0.9,
            "Huyện Củ Chi": 0.7,
            "Huyện Hóc Môn": 0.8,
            "Huyện Nhà Bè": 1.0,
            "Huyện Cần Giờ": 0.6,
        }

        factor = location_factors.get(features.area_name, 1.0)
        price = size * base_price_per_m2 * factor
        unit = "VND" if self.mode == "Sell" else "VND/tháng"
        print(f"Fallback prediction ({self.mode}): {price:,.0f} {unit} (size={size}m², factor={factor})")

        return price

    def predict(self, features: PropertyFeatures) -> Optional[float]:
        """
        Predict price using the trained model, with fallback to heuristic.

        Args:
            features: PropertyFeatures object

        Returns:
            Predicted price in VND, or None if insufficient data
        """
        print(f"PricePredictor.predict called for area: {features.area_name}")

        # Basic validation: need at least area_name and size info
        if not features.area_name:
            print("Prediction skipped: missing area_name")
            return None

        # Use fallback if model couldn't be loaded
        if self._use_fallback:
            print("Using fallback mode (trained model not available)")
            return self._fallback_predict(features)

        # Try trained model prediction
        print("Attempting prediction with trained model...")
        result = self._predictor.predict(features)

        # If model prediction fails, use fallback
        if result is None:
            print("Trained model prediction failed, falling back to heuristic")
            return self._fallback_predict(features)

        print(f"Final prediction result: {result:,.0f} VND")
        return result

    def predict_with_confidence(self, features: PropertyFeatures) -> Dict[str, Any]:
        """
        Predict price with confidence score using the trained model, with fallback to heuristic.

        Args:
            features: PropertyFeatures object

        Returns:
            Dictionary with predicted price, confidence score, and feature info.
            Returns error info if prediction fails.
        """
        print(f"PricePredictor.predict_with_confidence called for area: {features.area_name}")

        # Basic validation: need at least area_name and size info
        if not features.area_name:
            print("Prediction skipped: missing area_name")
            return {"predicted_price": None, "error": "Thiếu thông tin vị trí (quận/huyện)"}

        # Use fallback if model couldn't be loaded
        if self._use_fallback:
            print("Using fallback mode (trained model not available)")
            fallback_price = self._fallback_predict(features)
            return {
                "predicted_price": fallback_price,
                "log_price": None,
                "confidence": 0.50,  # Lower confidence for fallback model
                "features_used": {
                    "has_coordinates": features.latitude is not None and features.longitude is not None,
                    "has_size": True,
                    "area_name": features.area_name,
                },
                "shap_explanation": None,  # No SHAP for fallback
                "is_fallback": True,
            }

        # Try trained model prediction with confidence
        print("Attempting prediction with confidence using trained model...")
        result = self._predictor.predict_with_confidence(features)

        # If model prediction fails, use fallback
        if result.get("predicted_price") is None:
            print("Trained model prediction failed, falling back to heuristic")
            fallback_price = self._fallback_predict(features)
            return {
                "predicted_price": fallback_price,
                "log_price": None,
                "confidence": 0.50,  # Lower confidence for fallback model
                "features_used": {
                    "has_coordinates": features.latitude is not None and features.longitude is not None,
                    "has_size": True,
                    "area_name": features.area_name,
                },
                "shap_explanation": None,  # No SHAP for fallback
                "is_fallback": True,
            }

        result["is_fallback"] = False
        print(f"Final prediction with confidence: {result.get('predicted_price', 0):,.0f} VND (confidence: {result.get('confidence', 0):.1%})")
        if result.get("shap_explanation"):
            print("SHAP explanation included in result")
        return result
