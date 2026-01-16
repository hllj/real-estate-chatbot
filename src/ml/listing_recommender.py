"""
Listing Recommender Engine for Real Estate

This module provides smart recommendation of similar listings from the database.
Uses progressive search that relaxes criteria based on feature importance when
exact matches are not found.

Feature importance is derived from trained XGBoost models.
"""

import logging
import math
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd

from src.models import PropertyFeatures

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL pattern for nhatot.com listings
NHATOT_URL_PATTERN = "https://www.nhatot.com/{list_id}.htm"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH_SELL = PROJECT_ROOT / "data" / "s_listing_ingestion.csv"
DATA_PATH_RENT = PROJECT_ROOT / "data" / "u_listing_ingestion.csv"

# Aggregated feature importance from trained models
# Higher value = more important, should be relaxed LAST
# These values are aggregated from one-hot encoded feature importances
FEATURE_IMPORTANCE_SELL = {
    # Legal document is the most important! (0.22 from model)
    "property_legal_document_status": 100,

    # Location features
    "area_name": 85,

    # Property type features
    "house_type_name": 75,
    "apartment_type_name": 70,
    "category_name": 65,
    "commercial_type_name": 55,
    "land_type_name": 50,

    # Structure features
    "rooms_count": 80,
    "toilets_count": 45,
    "floors": 40,
    "floornumber": 35,

    # Size is important
    "log_size": 60,

    # Other features
    "direction_name": 25,
    "balconydirection_name": 20,
    "furnishing_sell_status": 15,
    "property_status_name": 10,
    "is_main_street": 5,
}

FEATURE_IMPORTANCE_RENT = {
    # Location features
    "area_name": 100,

    # Structure features - very important for rentals
    "rooms_count": 90,
    "category_name": 85,

    # Property type
    "house_type_name": 75,
    "apartment_type_name": 70,
    "commercial_type_name": 55,
    "land_type_name": 50,

    # Size
    "log_size": 65,

    # Other structure
    "toilets_count": 45,
    "floors": 40,
    "floornumber": 35,

    # Features
    "furnishing_rent_status": 30,
    "direction_name": 25,
    "balconydirection_name": 20,
    "deposit": 15,
    "is_good_room": 10,
    "is_main_street": 5,
}


class ListingRecommender:
    """
    Recommends similar listings from the database based on user criteria.
    Uses progressive search that relaxes criteria when no exact matches found.
    """

    def __init__(self, mode: str = "Sell"):
        """
        Initialize the recommender.

        Args:
            mode: "Sell" or "Rent" - determines which dataset to use
        """
        self.mode = mode
        self._df = None
        self.feature_importance = (
            FEATURE_IMPORTANCE_SELL if mode == "Sell" else FEATURE_IMPORTANCE_RENT
        )
        self.data_path = DATA_PATH_SELL if mode == "Sell" else DATA_PATH_RENT

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load the listing data."""
        if self._df is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            print(f"[ListingRecommender] Loading listing data from {self.data_path}")
            logger.info(f"Loading listing data from {self.data_path}")
            self._df = pd.read_csv(self.data_path)

            # Price is already in VND in ingestion files
            # Compute log_price and log_size for filtering
            self._df["log_price"] = self._df["price"].apply(lambda x: math.log10(x) if pd.notna(x) and x > 0 else None)
            self._df["log_size"] = self._df["size"].apply(lambda x: math.log10(x) if pd.notna(x) and x > 0 else None)

            # Generate URL from list_id
            self._df["url"] = self._df["list_id"].apply(
                lambda x: NHATOT_URL_PATTERN.format(list_id=int(x)) if pd.notna(x) else None
            )

            print(f"[ListingRecommender] Loaded {len(self._df)} listings")
            logger.info(f"Loaded {len(self._df)} listings with columns: {list(self._df.columns)}")
        return self._df

    def _get_sorted_features_for_relaxation(self) -> List[str]:
        """Get features sorted by importance (least important first for relaxation)."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=False  # Ascending - least important first
        )
        return [f[0] for f in sorted_features]

    def _features_to_query_dict(
        self,
        features: PropertyFeatures,
        predicted_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Convert PropertyFeatures to a dictionary for querying.

        Args:
            features: PropertyFeatures object
            predicted_price: Optional predicted price to use for price range

        Returns:
            Dictionary of feature values for matching
        """
        query = {}
        features_dict = features.model_dump(exclude_none=True)

        # Direct field mapping (PropertyFeatures field -> CSV column)
        direct_fields = [
            "area_name", "category_name", "floornumber", "is_main_street",
            "floors", "length", "living_size", "width", "apartment_type_name",
            "property_legal_document_status", "rooms_count", "toilets_count",
            "furnishing_sell_status", "furnishing_rent_status",
            "balconydirection_name", "direction_name", "house_type_name",
            "commercial_type_name", "land_type_name", "property_status_name",
            "deposit", "is_good_room"
        ]

        for field in direct_fields:
            if field in features_dict and features_dict[field] is not None:
                value = features_dict[field]
                # Skip "Không có thông tin" values
                if value != "Không có thông tin":
                    query[field] = value

        # Handle size - compute log_size from available size info
        size = features.size or features.living_size
        if size is None and features.width and features.length:
            size = features.width * features.length
        if size:
            query["log_size"] = math.log10(size)
            query["_size"] = size  # Keep raw size for display

        # Handle price range from predicted_price or actual_price
        target_price = predicted_price or features.actual_price
        if target_price:
            query["_target_price"] = target_price

        return query

    def _build_filter_mask(
        self,
        query: Dict[str, Any],
        features_to_use: List[str],
        price_range_pct: float = 0.2,
        size_range_pct: float = 0.3
    ) -> pd.Series:
        """
        Build a boolean mask for filtering listings.

        Args:
            query: Query dictionary with feature values
            features_to_use: List of features to match on
            price_range_pct: Price range tolerance (e.g., 0.2 = ±20%)
            size_range_pct: Size range tolerance

        Returns:
            Boolean Series for filtering DataFrame
        """
        df = self.df
        mask = pd.Series([True] * len(df), index=df.index)

        for feature in features_to_use:
            if feature not in query:
                continue

            value = query[feature]

            if feature == "log_size" and value is not None:
                # Size range filter using log scale
                log_lower = value - math.log10(1 + size_range_pct)
                log_upper = value + math.log10(1 + size_range_pct)
                mask &= (df["log_size"] >= log_lower) & (df["log_size"] <= log_upper)

            elif feature in ["floornumber", "floors"]:
                # Numeric features - allow some tolerance
                if pd.notna(value):
                    col_values = pd.to_numeric(df[feature], errors='coerce')
                    tolerance = max(1, abs(value) * 0.3)  # 30% tolerance or at least 1
                    mask &= ((col_values >= value - tolerance) & (col_values <= value + tolerance)) | col_values.isna()

            elif feature == "deposit":
                # Deposit with wider tolerance
                if pd.notna(value) and value > 0:
                    col_values = pd.to_numeric(df[feature], errors='coerce')
                    tolerance = value * 0.5  # 50% tolerance
                    mask &= ((col_values >= value - tolerance) & (col_values <= value + tolerance)) | col_values.isna()

            elif feature in ["is_main_street", "is_good_room"]:
                # Boolean features - convert to string comparison
                if value is not None:
                    df_col = df[feature].astype(str).str.lower()
                    query_val = str(value).lower()
                    # Allow match or missing
                    mask &= (df_col == query_val) | (df_col == 'nan') | (df_col == '')

            elif feature in df.columns:
                # Categorical features - exact match or allow "Không có thông tin"
                df_col = df[feature].fillna("Không có thông tin")
                mask &= (df_col == value) | (df_col == "Không có thông tin")

        # Apply price filter if target price exists
        if "_target_price" in query:
            target = query["_target_price"]
            log_target = math.log10(target)
            log_lower = log_target - math.log10(1 + price_range_pct)
            log_upper = log_target + math.log10(1 + price_range_pct)
            mask &= (df["log_price"] >= log_lower) & (df["log_price"] <= log_upper)

        return mask

    def _calculate_similarity_score(
        self,
        row: pd.Series,
        query: Dict[str, Any],
        features_to_use: List[str]
    ) -> float:
        """
        Calculate similarity score between a listing and the query.

        Args:
            row: DataFrame row representing a listing
            query: Query dictionary
            features_to_use: Features being used for matching

        Returns:
            Similarity score (0-100)
        """
        total_weight = 0
        matched_weight = 0

        for feature in features_to_use:
            if feature not in query or feature.startswith("_"):
                continue

            importance = self.feature_importance.get(feature, 10)
            total_weight += importance

            query_val = query.get(feature)
            row_val = row.get(feature)

            if query_val is None:
                continue

            if feature == "log_size":
                # Size similarity - closer is better
                if pd.notna(row_val):
                    diff = abs(query_val - row_val)
                    if diff < 0.5:  # Within ~3x size range
                        score = max(0, 1 - diff / 0.5)
                        matched_weight += importance * score
            elif feature in ["floornumber", "floors"]:
                # Numeric similarity
                if pd.notna(row_val):
                    try:
                        row_num = float(row_val)
                        diff = abs(query_val - row_num)
                        max_diff = max(query_val * 0.5, 2)
                        if diff <= max_diff:
                            score = max(0, 1 - diff / max_diff)
                            matched_weight += importance * score
                    except (ValueError, TypeError):
                        pass
            else:
                # Categorical exact match
                if str(query_val) == str(row_val):
                    matched_weight += importance
                elif str(row_val) == "Không có thông tin":
                    matched_weight += importance * 0.3  # Partial credit for missing

        # Add price similarity bonus
        if "_target_price" in query and "log_price" in row:
            target_log = math.log10(query["_target_price"])
            row_log = row.get("log_price", 0)
            if pd.notna(row_log):
                diff = abs(target_log - row_log)
                if diff < 0.5:
                    price_bonus = (1 - diff / 0.5) * 20  # Up to 20 bonus points
                    matched_weight += price_bonus
                    total_weight += 20

        if total_weight == 0:
            return 0

        return min(100, (matched_weight / total_weight) * 100)

    def search_listings(
        self,
        features: PropertyFeatures,
        predicted_price: Optional[float] = None,
        max_results: int = 5,
        min_similarity: float = 20.0,
        initial_price_range: float = 0.15,
        max_price_range: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search for similar listings with progressive criteria relaxation.

        Args:
            features: PropertyFeatures to match
            predicted_price: Optional predicted price for price range filtering
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity score to include (0-100)
            initial_price_range: Initial price range tolerance (e.g., 0.15 = ±15%)
            max_price_range: Maximum price range to expand to

        Returns:
            Dictionary with search results and metadata
        """
        print(f"\n[ListingRecommender] === Starting search (mode={self.mode}) ===")
        print(f"[ListingRecommender] Features: {features.model_dump(exclude_none=True)}")
        print(f"[ListingRecommender] Predicted price: {predicted_price}")

        try:
            query = self._features_to_query_dict(features, predicted_price)
            print(f"[ListingRecommender] Query dict: {query}")
        except Exception as e:
            print(f"[ListingRecommender] ERROR converting features to query: {e}")
            logger.error(f"Error converting features to query: {e}")
            return {
                "success": False,
                "error": str(e),
                "listings": [],
                "message": f"Lỗi khi xử lý tiêu chí tìm kiếm: {e}"
            }

        # Get features sorted by importance (least important first for relaxation)
        all_features = self._get_sorted_features_for_relaxation()
        print(f"[ListingRecommender] Feature relaxation order: {all_features}")

        # Filter to only features that exist in query
        available_features = [f for f in all_features if f in query]
        print(f"[ListingRecommender] Available features in query: {available_features}")

        if not available_features:
            print("[ListingRecommender] No search criteria provided")
            return {
                "success": False,
                "error": "No search criteria provided",
                "listings": [],
                "message": "Vui lòng cung cấp ít nhất một tiêu chí tìm kiếm (khu vực, loại BĐS, diện tích, v.v.)"
            }

        results = []
        relaxation_log = []
        current_price_range = initial_price_range
        features_removed = []
        features_to_use = available_features.copy()

        # Progressive search
        max_iterations = len(available_features) + 10
        iteration = 0

        print(f"[ListingRecommender] Starting progressive search...")

        while iteration < max_iterations:
            iteration += 1

            # Build filter
            try:
                mask = self._build_filter_mask(
                    query,
                    features_to_use,
                    price_range_pct=current_price_range
                )
            except Exception as e:
                print(f"[ListingRecommender] ERROR building filter: {e}")
                logger.error(f"Error building filter: {e}")
                break

            filtered_df = self.df[mask]
            print(f"[ListingRecommender] Iteration {iteration}: Found {len(filtered_df)} candidates (price range ±{int(current_price_range*100)}%, features: {len(features_to_use)})")

            # Score and collect results
            for idx, row in filtered_df.iterrows():
                # Skip if already in results
                if any(r.get("_idx") == idx for r in results):
                    continue

                score = self._calculate_similarity_score(row, query, features_to_use)
                if score >= min_similarity:
                    results.append({
                        "_idx": idx,
                        "list_id": row.get("list_id"),
                        "subject": row.get("subject"),
                        "url": row.get("url"),
                        "area_name": row.get("area_name"),
                        "category_name": row.get("category_name"),
                        "price": row.get("price"),
                        "size": row.get("size"),
                        "rooms_count": row.get("rooms_count"),
                        "toilets_count": row.get("toilets_count"),
                        "floors": row.get("floors"),
                        "floornumber": row.get("floornumber"),
                        "direction_name": row.get("direction_name"),
                        "house_type_name": row.get("house_type_name"),
                        "apartment_type_name": row.get("apartment_type_name"),
                        "property_legal_document_status": row.get("property_legal_document_status"),
                        "furnishing_status": row.get("furnishing_sell_status") or row.get("furnishing_rent_status"),
                        "similarity_score": round(score, 1),
                        "features_relaxed": features_removed.copy()
                    })

            # Sort by similarity
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Keep only top results
            if len(results) >= max_results * 2:
                results = results[:max_results * 2]

            # Check if we have enough good results
            good_results = [r for r in results if r["similarity_score"] >= min_similarity]
            if len(good_results) >= max_results:
                break

            # Relax criteria
            if current_price_range < max_price_range:
                current_price_range += 0.1
                relaxation_log.append(f"Mở rộng khoảng giá ±{int(current_price_range*100)}%")
                print(f"[ListingRecommender] Relaxing: Expanding price range to ±{int(current_price_range*100)}%")
            elif features_to_use:
                removed = features_to_use.pop(0)  # Remove least important
                features_removed.append(removed)
                relaxation_log.append(f"Bỏ qua: {self._get_feature_name_vn(removed)}")
                print(f"[ListingRecommender] Relaxing: Removing feature '{removed}' ({self._get_feature_name_vn(removed)})")
            else:
                print(f"[ListingRecommender] No more criteria to relax")
                break

        # Final filtering and formatting
        final_results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:max_results]
        print(f"[ListingRecommender] Final results: {len(final_results)} listings")

        formatted_results = []
        for r in final_results:
            formatted = {
                "list_id": r.get("list_id"),
                "subject": r.get("subject") if pd.notna(r.get("subject")) else "N/A",
                "url": r.get("url") if pd.notna(r.get("url")) else None,
                "khu_vuc": r["area_name"],
                "loai_bds": r["category_name"],
                "gia": self._format_price(r["price"]),
                "gia_raw": r["price"],
                "dien_tich": f"{r['size']:.1f} m²" if r["size"] and pd.notna(r["size"]) else "N/A",
                "dien_tich_raw": r["size"] if pd.notna(r.get("size")) else None,
                "so_phong_ngu": str(r["rooms_count"]) if pd.notna(r.get("rooms_count")) else "N/A",
                "so_toilet": str(r["toilets_count"]) if pd.notna(r.get("toilets_count")) else "N/A",
                "so_tang": str(int(r["floors"])) if pd.notna(r.get("floors")) else "N/A",
                "tang_so": str(int(r["floornumber"])) if pd.notna(r.get("floornumber")) else "N/A",
                "huong": r.get("direction_name") if pd.notna(r.get("direction_name")) and r.get("direction_name") != "Không có thông tin" else "N/A",
                "loai_nha": self._get_property_subtype(r),
                "phap_ly": r.get("property_legal_document_status") if pd.notna(r.get("property_legal_document_status")) and r.get("property_legal_document_status") != "Không có thông tin" else "N/A",
                "noi_that": r.get("furnishing_status") if pd.notna(r.get("furnishing_status")) and r.get("furnishing_status") != "Không có thông tin" else "N/A",
                "do_tuong_dong": f"{r['similarity_score']}%",
                "similarity_score": r["similarity_score"],
            }
            formatted_results.append(formatted)
            print(f"[ListingRecommender] Found listing: {formatted['subject'][:50]}... | {formatted['gia']} | {formatted['do_tuong_dong']}")

        # Build search criteria summary for response
        search_criteria = {
            "khu_vuc": query.get("area_name"),
            "loai_bds": query.get("category_name"),
            "gia_muc_tieu": self._format_price(query.get("_target_price")) if query.get("_target_price") else None,
            "dien_tich": f"{query.get('_size'):.1f} m²" if query.get("_size") else None,
            "so_phong_ngu": query.get("rooms_count"),
        }
        # Remove None values
        search_criteria = {k: v for k, v in search_criteria.items() if v is not None}

        return {
            "success": len(formatted_results) > 0,
            "mode": self.mode,
            "total_found": len(formatted_results),
            "listings": formatted_results,
            "search_criteria": search_criteria,
            "relaxation_applied": relaxation_log if relaxation_log else None,
            "final_price_range_pct": int(current_price_range * 100),
            "message": self._generate_summary_message(formatted_results, relaxation_log, search_criteria)
        }

    def _get_property_subtype(self, row: Dict) -> str:
        """Get the specific property subtype name."""
        subtypes = [
            row.get("house_type_name"),
            row.get("apartment_type_name"),
            row.get("commercial_type_name"),
            row.get("land_type_name")
        ]
        for st in subtypes:
            if st and pd.notna(st) and st != "Không có thông tin":
                return st
        return "N/A"

    def _get_feature_name_vn(self, feature: str) -> str:
        """Get Vietnamese name for a feature."""
        names = {
            "area_name": "Khu vực",
            "category_name": "Loại BĐS",
            "house_type_name": "Loại nhà",
            "apartment_type_name": "Loại căn hộ",
            "commercial_type_name": "Loại thương mại",
            "land_type_name": "Loại đất",
            "rooms_count": "Số phòng ngủ",
            "toilets_count": "Số toilet",
            "floors": "Số tầng",
            "floornumber": "Tầng số",
            "log_size": "Diện tích",
            "direction_name": "Hướng nhà",
            "balconydirection_name": "Hướng ban công",
            "furnishing_sell_status": "Nội thất",
            "furnishing_rent_status": "Nội thất",
            "property_legal_document_status": "Pháp lý",
            "property_status_name": "Tình trạng",
            "is_main_street": "Mặt tiền",
            "deposit": "Tiền cọc",
            "is_good_room": "Đánh giá phòng",
        }
        return names.get(feature, feature)

    def _format_price(self, price: float) -> str:
        """Format price in Vietnamese style."""
        if price is None or pd.isna(price):
            return "N/A"

        if self.mode == "Sell":
            if price >= 1_000_000_000:
                return f"{price / 1_000_000_000:.2f} tỷ"
            else:
                return f"{price / 1_000_000:.0f} triệu"
        else:  # Rent
            if price >= 1_000_000:
                return f"{price / 1_000_000:.1f} triệu/tháng"
            else:
                return f"{price / 1_000:.0f} nghìn/tháng"

    def _generate_summary_message(
        self,
        results: List[Dict],
        relaxation_log: List[str],
        search_criteria: Dict[str, Any]
    ) -> str:
        """Generate a Vietnamese summary message."""
        if not results:
            return "Không tìm thấy bất động sản nào phù hợp với tiêu chí tìm kiếm."

        mode_text = "bán" if self.mode == "Sell" else "cho thuê"

        # Build criteria summary
        criteria_parts = []
        if search_criteria.get("khu_vuc"):
            criteria_parts.append(f"tại {search_criteria['khu_vuc']}")
        if search_criteria.get("loai_bds"):
            criteria_parts.append(f"loại {search_criteria['loai_bds']}")
        if search_criteria.get("dien_tich"):
            criteria_parts.append(f"diện tích ~{search_criteria['dien_tich']}")
        if search_criteria.get("gia_muc_tieu"):
            criteria_parts.append(f"giá ~{search_criteria['gia_muc_tieu']}")

        criteria_str = ", ".join(criteria_parts) if criteria_parts else ""

        msg = f"Tìm thấy {len(results)} bất động sản {mode_text} tương tự"
        if criteria_str:
            msg += f" ({criteria_str})"
        msg += ":\n"

        if relaxation_log:
            msg += f"\n📝 Đã điều chỉnh tiêu chí: {', '.join(relaxation_log)}\n"

        for i, r in enumerate(results, 1):
            # Show subject/title first
            subject = r.get('subject', 'N/A')
            if subject != "N/A" and len(subject) > 80:
                subject = subject[:80] + "..."
            msg += f"\n**{i}. {subject}**"
            msg += f"\n   📍 {r['loai_bds']} tại {r['khu_vuc']}"
            msg += f"\n   💰 Giá: {r['gia']}"
            msg += f"\n   📐 Diện tích: {r['dien_tich']}"
            if r['so_phong_ngu'] != "N/A":
                msg += f" | 🛏️ {r['so_phong_ngu']} PN"
            if r['so_toilet'] != "N/A":
                msg += f" | 🚿 {r['so_toilet']} WC"
            if r['loai_nha'] != "N/A":
                msg += f"\n   🏠 Loại: {r['loai_nha']}"
            if r['phap_ly'] != "N/A":
                msg += f" | 📋 {r['phap_ly']}"
            msg += f"\n   ⭐ Độ tương đồng: **{r['do_tuong_dong']}**"
            # Add URL if available
            if r.get('url'):
                msg += f"\n   🔗 Link: {r['url']}"

        return msg


# Singleton instances
_recommender_instances: Dict[str, ListingRecommender] = {}


def get_recommender(mode: str = "Sell") -> ListingRecommender:
    """Get or create singleton recommender instance."""
    global _recommender_instances
    if mode not in _recommender_instances:
        _recommender_instances[mode] = ListingRecommender(mode=mode)
    return _recommender_instances[mode]
