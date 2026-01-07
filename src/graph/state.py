from typing import Annotated, TypedDict, List, Dict, Any, Optional, Tuple
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from src.models import PropertyFeatures


class PredictionResult(TypedDict, total=False):
    """
    Kết quả dự đoán giá với thông tin confidence.
    """
    predicted_price: Optional[float]
    log_price: Optional[float]
    confidence_interval_90: Optional[Tuple[float, float]]  # (lower, upper) - 90% CI from tree estimators
    features_used: Optional[Dict[str, Any]]
    error: Optional[str]


class GraphState(TypedDict):
    """
    Trạng thái của đồ thị quy trình (Graph State).
    """
    messages: Annotated[List[BaseMessage], add_messages]
    features: PropertyFeatures
    user_input_url: Optional[str]
    prediction_result: Optional[PredictionResult]  # Current prediction with confidence
    previous_prediction: Optional[PredictionResult]  # Previous prediction for comparison
    unknown_fields: List[str]  # Track fields user explicitly doesn't know
