from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from src.models import PropertyFeatures

class GraphState(TypedDict):
    """
    Trạng thái của đồ thị quy trình (Graph State).
    """
    messages: Annotated[List[BaseMessage], add_messages]
    features: PropertyFeatures
    user_input_url: Optional[str]
    prediction_result: Optional[float]
