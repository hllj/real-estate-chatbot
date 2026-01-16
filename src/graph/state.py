from typing import Annotated, TypedDict, List, Dict, Any, Optional, Tuple, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from src.models import PropertyFeatures, ModeType


class ShapFeatureContribution(TypedDict, total=False):
    """
    SHAP contribution for a single feature.
    """
    feature: str  # Feature name (technical)
    feature_vn: str  # Feature name in Vietnamese
    shap_value: float  # SHAP value (in log scale)
    feature_value: Optional[Any]  # Actual feature value
    impact: str  # "positive" or "negative"


class ShapExplanation(TypedDict, total=False):
    """
    SHAP explanation for a prediction.
    """
    success: bool
    base_value: Optional[float]  # Expected model output (mean prediction)
    predicted_log_price: Optional[float]
    top_features: Optional[List[ShapFeatureContribution]]
    all_contributions: Optional[List[ShapFeatureContribution]]
    error: Optional[str]


class PredictionResult(TypedDict, total=False):
    """
    Kết quả dự đoán giá với thông tin confidence và SHAP explanation.
    """
    predicted_price: Optional[float]
    log_price: Optional[float]
    confidence_interval_90: Optional[Tuple[float, float]]  # (lower, upper) - 90% CI from tree estimators
    features_used: Optional[Dict[str, Any]]
    shap_explanation: Optional[ShapExplanation]  # SHAP-based price breakdown
    is_fallback: Optional[bool]  # True if using fallback heuristic model
    error: Optional[str]


class PriceComparison(TypedDict, total=False):
    """
    So sánh giữa giá dự đoán và giá thực tế.
    """
    predicted_price: float  # Giá dự đoán (VNĐ)
    actual_price: float  # Giá thực tế (VNĐ)
    difference: float  # Chênh lệch (actual - predicted)
    difference_percent: float  # Phần trăm chênh lệch so với giá thực tế
    accuracy_level: str  # "Xuất sắc" (<10%), "Tốt" (<20%), "Khá" (<30%), "Cần cải thiện" (>30%)
    comparison_text_vn: str  # Giải thích bằng tiếng Việt


class ListingRecommendation(TypedDict, total=False):
    """
    Kết quả tìm kiếm bất động sản tương tự.
    """
    list_id: Optional[int]  # ID tin đăng trên nhatot.com
    subject: str  # Tiêu đề tin đăng
    url: Optional[str]  # URL link đến tin đăng (nhatot.com/{list_id}.htm)
    khu_vuc: str  # Quận/Huyện
    loai_bds: str  # Loại BĐS
    gia: str  # Giá đã format (vd: "5.5 tỷ")
    gia_raw: float  # Giá thô (VNĐ)
    dien_tich: str  # Diện tích đã format (vd: "100 m²")
    dien_tich_raw: Optional[float]  # Diện tích thô (m²)
    so_phong_ngu: str  # Số phòng ngủ
    so_toilet: str  # Số toilet
    so_tang: str  # Số tầng
    tang_so: str  # Tầng số (cho chung cư)
    huong: str  # Hướng nhà
    loai_nha: str  # Loại nhà/căn hộ cụ thể
    phap_ly: str  # Tình trạng pháp lý
    noi_that: str  # Tình trạng nội thất
    do_tuong_dong: str  # Độ tương đồng (vd: "85%")
    similarity_score: float  # Điểm tương đồng (0-100)


class ListingSearchResult(TypedDict, total=False):
    """
    Kết quả tìm kiếm bất động sản tương tự.
    """
    success: bool  # Tìm kiếm thành công hay không
    mode: str  # "Sell" hoặc "Rent"
    total_found: int  # Số lượng BĐS tìm được
    listings: List[ListingRecommendation]  # Danh sách BĐS
    search_criteria: Dict[str, Any]  # Tiêu chí tìm kiếm
    relaxation_applied: Optional[List[str]]  # Các tiêu chí đã nới lỏng
    final_price_range_pct: int  # Khoảng giá cuối cùng (%)
    message: str  # Thông báo tóm tắt


class GraphState(TypedDict):
    """
    Trạng thái của đồ thị quy trình (Graph State).
    """
    messages: Annotated[List[BaseMessage], add_messages]
    features: PropertyFeatures
    mode: ModeType  # "Sell" or "Rent" - determines which model to use
    user_input_url: Optional[str]
    prediction_result: Optional[PredictionResult]  # Current prediction with confidence
    previous_prediction: Optional[PredictionResult]  # Previous prediction for comparison
    unknown_fields: List[str]  # Track fields user explicitly doesn't know
    price_comparison: Optional[PriceComparison]  # Comparison between predicted and actual price
    listing_recommendations: Optional[ListingSearchResult]  # Similar listings from database
