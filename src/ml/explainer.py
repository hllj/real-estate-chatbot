"""
SHAP-based explainer for real estate price predictions.

This module provides interpretable explanations for property price predictions
using SHAP (SHapley Additive exPlanations) values.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Vietnamese translations for feature names
FEATURE_NAMES_VN = {
    # Location features
    "area_name": "Quận/Huyện",
    "longitude": "Kinh độ",
    "latitude": "Vĩ độ",
    "dist_center_km": "Khoảng cách đến trung tâm TP",

    # Size features
    "log_size": "Diện tích",
    "size": "Diện tích",
    "living_size": "Diện tích sử dụng",
    "width": "Chiều ngang",
    "length": "Chiều dài",

    # Property type
    "category_name": "Loại bất động sản",
    "house_type_name": "Loại nhà",
    "apartment_type_name": "Loại căn hộ",
    "commercial_type_name": "Loại thương mại",
    "land_type_name": "Loại đất",

    # Structure
    "floors": "Số tầng",
    "floornumber": "Tầng số",
    "rooms_count": "Số phòng ngủ",
    "toilets_count": "Số phòng vệ sinh",

    # Features
    "direction_name": "Hướng nhà",
    "balconydirection_name": "Hướng ban công",
    "furnishing_sell_status": "Tình trạng nội thất",
    "is_main_street": "Mặt tiền đường lớn",

    # Legal
    "property_legal_document_status": "Pháp lý",
    "property_status_name": "Tình trạng bàn giao",

    # Amenity distances
    "dist_min_metro_km": "Khoảng cách đến Metro",
    "dist_min_cho_km": "Khoảng cách đến chợ",
    "dist_min_sieu_thi_km": "Khoảng cách đến siêu thị",
    "dist_min_sieu_thi_mini_km": "Khoảng cách đến siêu thị mini",
    "dist_min_cua_hang_tien_loi_km": "Khoảng cách đến cửa hàng tiện lợi",
    "dist_min_truong_mam_non_km": "Khoảng cách đến trường mầm non",
    "dist_min_truong_tieu_hoc_km": "Khoảng cách đến trường tiểu học",
    "dist_min_truong_thpt_km": "Khoảng cách đến trường THPT",
    "dist_min_truong_ai_hoc_km": "Khoảng cách đến đại học",
    "dist_min_benh_vien_km": "Khoảng cách đến bệnh viện",
    "dist_min_tram_y_te_km": "Khoảng cách đến trạm y tế",
    "dist_min_cong_vien_km": "Khoảng cách đến công viên",
    "dist_min_san_van_ong_km": "Khoảng cách đến sân vận động",
    "dist_min_khu_vui_choi_km": "Khoảng cách đến khu vui chơi",
    "dist_min_pho_i_bo_km": "Khoảng cách đến phố đi bộ",
    "dist_min_quan_an_km": "Khoảng cách đến quán ăn",
    "dist_min_nha_hang_km": "Khoảng cách đến nhà hàng",
    "dist_min_quan_cafe_km": "Khoảng cách đến quán cafe",
    "dist_min_trung_tam_thuong_mai_km": "Khoảng cách đến TTTM",
    "dist_min_ngan_hang_km": "Khoảng cách đến ngân hàng",
    "dist_min_atm_km": "Khoảng cách đến ATM",
    "dist_min_ben_xe_km": "Khoảng cách đến bến xe",
    "dist_min_ga_tau_km": "Khoảng cách đến ga tàu",

    # Amenity counts
    "amenities_within_300m": "Tiện ích trong 300m",
    "amenities_within_500m": "Tiện ích trong 500m",
    "amenities_within_1000m": "Tiện ích trong 1km",
    "amenities_within_5000m": "Tiện ích trong 5km",
}


class ShapExplainer:
    """
    SHAP-based explainer for real estate price predictions.

    Uses TreeExplainer for efficient computation on tree-based models
    (XGBoost, LightGBM, Random Forest, etc.)
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained sklearn pipeline or model
            feature_names: List of feature names (before one-hot encoding)
        """
        self._model = model
        self._feature_names = feature_names
        self._explainer = None
        self._preprocessor = None
        self._encoded_feature_names = None

        # Extract preprocessor and model from pipeline
        if hasattr(model, 'named_steps'):
            self._preprocessor = model.named_steps.get('preprocessor')
            self._estimator = model.named_steps.get('model')
        else:
            self._estimator = model

    @property
    def explainer(self):
        """Lazy load TreeExplainer on first access."""
        if self._explainer is None:
            try:
                import shap
                logger.info("Initializing SHAP explainer...")

                # Try to fix XGBoost 2.x base_score compatibility issue
                estimator = self._estimator
                if hasattr(estimator, 'get_booster'):
                    # This is an XGBoost model - fix base_score if needed
                    estimator = self._fix_xgboost_base_score(estimator)

                # Try TreeExplainer first (fastest for tree models)
                try:
                    self._explainer = shap.TreeExplainer(estimator)
                    logger.info("SHAP TreeExplainer initialized successfully")
                except Exception as tree_error:
                    logger.warning(f"TreeExplainer failed: {tree_error}")
                    # Fallback to general Explainer (slower but more compatible)
                    logger.info("Falling back to shap.Explainer...")
                    self._explainer = shap.Explainer(estimator)
                    logger.info("SHAP Explainer initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {e}")
                raise
        return self._explainer

    def _fix_xgboost_base_score(self, model):
        """
        Fix XGBoost 2.x base_score compatibility issue with SHAP.

        XGBoost 2.x stores base_score as '[value]' string, which older SHAP
        versions cannot parse. This method patches the model config.
        """
        try:
            import json

            booster = model.get_booster()
            config = json.loads(booster.save_config())

            # Navigate to learner_model_param
            learner = config.get('learner', {})
            learner_model_param = learner.get('learner_model_param', {})
            base_score = learner_model_param.get('base_score', None)

            if base_score and isinstance(base_score, str) and base_score.startswith('['):
                # Parse the array format: '[9.762936E0]' -> 9.762936
                cleaned = base_score.strip('[]')
                # Handle scientific notation like '9.762936E0'
                new_base_score = str(float(cleaned))
                learner_model_param['base_score'] = new_base_score
                logger.info(f"Fixed XGBoost base_score: {base_score} -> {new_base_score}")

                # Save the fixed config back
                booster.load_config(json.dumps(config))

        except Exception as e:
            logger.warning(f"Could not fix XGBoost base_score: {e}")

        return model

    def _get_encoded_feature_names(self, X_processed: np.ndarray) -> List[str]:
        """Get feature names after preprocessing (including one-hot encoded)."""
        if self._encoded_feature_names is not None:
            return self._encoded_feature_names

        if self._preprocessor is not None:
            try:
                # Try to get feature names from preprocessor
                if hasattr(self._preprocessor, 'get_feature_names_out'):
                    self._encoded_feature_names = list(self._preprocessor.get_feature_names_out())
                else:
                    # Fallback: generate generic names
                    self._encoded_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            except Exception:
                self._encoded_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        else:
            self._encoded_feature_names = self._feature_names

        return self._encoded_feature_names

    def _aggregate_categorical_shap(
        self,
        shap_values: np.ndarray,
        encoded_names: List[str]
    ) -> Dict[str, float]:
        """
        Aggregate SHAP values for one-hot encoded categorical features.

        For example, area_name_Quận 1, area_name_Quận 7, etc. are aggregated
        into a single area_name contribution.
        """
        aggregated = {}

        for i, name in enumerate(encoded_names):
            # Safely extract scalar value from numpy array
            raw_val = shap_values[i]
            if isinstance(raw_val, np.ndarray):
                raw_val = raw_val.flatten()[0] if raw_val.size > 0 else 0.0
            if isinstance(raw_val, np.generic):
                shap_val = float(raw_val.item())
            else:
                shap_val = float(raw_val)

            # Check if this is a one-hot encoded feature (contains __)
            if '__' in name:
                # Extract base feature name (e.g., "cat__area_name_Quận 1" -> "area_name")
                parts = name.split('__')
                if len(parts) >= 2:
                    # Get the category part and extract base name
                    cat_part = parts[1]  # e.g., "area_name_Quận 1"
                    # Find the base feature name
                    for base_name in self._feature_names:
                        if cat_part.startswith(base_name):
                            if base_name not in aggregated:
                                aggregated[base_name] = 0.0
                            aggregated[base_name] += shap_val
                            break
                    else:
                        # Couldn't match, use as-is
                        aggregated[name] = shap_val
                else:
                    aggregated[name] = shap_val
            elif name.startswith('num__') or name.startswith('remainder__'):
                # Numeric feature from ColumnTransformer
                base_name = name.split('__')[1] if '__' in name else name
                aggregated[base_name] = shap_val
            else:
                # Regular feature name
                aggregated[name] = shap_val

        return aggregated

    def _to_scalar(self, value) -> float:
        """
        Safely convert a value to a Python float scalar.

        Handles numpy arrays, lists, and other array-like objects.
        """
        # If it's already a Python scalar, return it
        if isinstance(value, (int, float)) and not isinstance(value, np.generic):
            return float(value)

        # Handle numpy scalars
        if isinstance(value, np.generic):
            return float(value.item())

        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            # Flatten and get first element
            flat = value.flatten()
            if len(flat) > 0:
                return float(flat[0].item() if isinstance(flat[0], np.generic) else flat[0])
            return 0.0

        # Handle lists
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                return self._to_scalar(value[0])
            return 0.0

        # Try direct conversion as last resort
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(f"Could not convert {type(value)} to scalar: {value}")
            return 0.0

    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute SHAP values and return contributing features.

        Args:
            X: Input DataFrame with features (before preprocessing)
            top_n: Number of top features to return. None or -1 means all features.

        Returns:
            Dictionary with:
                - base_value: Expected model output (mean prediction)
                - predicted_log_price: The model's prediction
                - top_features: List of top contributing features (or all if top_n is None/-1)
                - all_contributions: All aggregated feature contributions
        """
        try:
            # Preprocess features if preprocessor exists
            if self._preprocessor is not None:
                X_processed = self._preprocessor.transform(X)
            else:
                X_processed = X.values if hasattr(X, 'values') else X

            # Get encoded feature names
            encoded_names = self._get_encoded_feature_names(X_processed)

            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_processed)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-output model
                shap_values = shap_values[0]

            # Ensure shap_values is a numpy array
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)

            # Get first sample if 2D
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # Flatten to 1D array
            shap_values = shap_values.flatten()

            # Get base value (expected value) and convert to scalar
            base_value = self._to_scalar(self.explainer.expected_value)

            # Aggregate SHAP values for categorical features
            aggregated_shap = self._aggregate_categorical_shap(shap_values, encoded_names)

            # Create list of contributions with Vietnamese names
            contributions = []
            for feature_name, shap_val in aggregated_shap.items():
                # Get Vietnamese name
                vn_name = FEATURE_NAMES_VN.get(feature_name, feature_name)

                # Get feature value from original DataFrame
                feature_value = None
                if feature_name in X.columns:
                    feature_value = X[feature_name].iloc[0]
                    # Convert to native Python type
                    if pd.isna(feature_value):
                        feature_value = None
                    elif isinstance(feature_value, np.generic):
                        feature_value = feature_value.item()
                    elif hasattr(feature_value, 'item'):
                        feature_value = feature_value.item()

                contributions.append({
                    'feature': feature_name,
                    'feature_vn': vn_name,
                    'shap_value': float(shap_val),
                    'feature_value': feature_value,
                    # Impact direction for display
                    'impact': 'positive' if shap_val > 0 else 'negative'
                })

            # Sort by absolute SHAP value (most impactful first)
            contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            # Compute predicted log_price from base + sum of SHAP values
            predicted_log_price = base_value + sum(aggregated_shap.values())

            # Handle top_n: None or -1 means all features
            if top_n is None or top_n < 0:
                top_features = contributions  # All features
            else:
                top_features = contributions[:top_n]

            return {
                'base_value': base_value,
                'predicted_log_price': predicted_log_price,
                'top_features': top_features,
                'all_contributions': contributions,
                'success': True
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }

    def format_explanation_text(
        self,
        explanation: Dict[str, Any],
        predicted_price: float,
        top_n: int = 5
    ) -> str:
        """
        Format SHAP explanation as human-readable Vietnamese text.

        Args:
            explanation: Output from explain_prediction()
            predicted_price: The predicted price in VND
            top_n: Number of features to include in text

        Returns:
            Formatted explanation string in Vietnamese
        """
        if not explanation.get('success'):
            return ""

        lines = []
        lines.append("**Phân tích các yếu tố ảnh hưởng đến giá:**\n")

        top_features = explanation.get('top_features', [])[:top_n]

        # Separate positive and negative impacts
        positive_impacts = [f for f in top_features if f['shap_value'] > 0.01]
        negative_impacts = [f for f in top_features if f['shap_value'] < -0.01]

        if positive_impacts:
            lines.append("📈 **Yếu tố làm TĂNG giá:**")
            for feat in positive_impacts[:3]:
                # Convert SHAP value in log scale to approximate percentage
                # SHAP value of 0.1 in log10 scale ≈ 26% increase
                pct_impact = (10 ** feat['shap_value'] - 1) * 100
                value_str = ""
                if feat['feature_value'] is not None:
                    value_str = f" ({feat['feature_value']})"
                lines.append(f"  • {feat['feature_vn']}{value_str}: +{pct_impact:.0f}%")

        if negative_impacts:
            lines.append("\n📉 **Yếu tố làm GIẢM giá:**")
            for feat in negative_impacts[:3]:
                pct_impact = (1 - 10 ** feat['shap_value']) * 100
                value_str = ""
                if feat['feature_value'] is not None:
                    value_str = f" ({feat['feature_value']})"
                lines.append(f"  • {feat['feature_vn']}{value_str}: -{pct_impact:.0f}%")

        return "\n".join(lines)


def create_explainer(model, feature_names: List[str]) -> Optional[ShapExplainer]:
    """
    Factory function to create a SHAP explainer.

    Returns None if SHAP is not available or initialization fails.
    """
    try:
        import shap
        return ShapExplainer(model, feature_names)
    except ImportError:
        logger.warning("SHAP library not installed. Explanations will not be available.")
        return None
    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        return None
