import os
import re
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from dotenv import load_dotenv

from src.graph.state import GraphState
from src.models import PropertyFeatures
from src.ml.placeholder_model import PricePredictor
from src.utils.scraper import fetch_property_details

load_dotenv()

# Setup LLM
llm = ChatGoogleGenerativeAI(model=os.environ["GEMINI_MODEL"], temperature=0, convert_system_message_to_human=True)

# Vietnamese System Prompt
SYSTEM_PROMPT = """Bạn là một trợ lý thông minh chuyên về bất động sản tại Việt Nam.
Nhiệm vụ của bạn là thu thập thông tin từ người dùng để dự đoán giá nhà.
Hãy giao tiếp bằng tiếng Việt một cách tự nhiên, chuyên nghiệp và thân thiện.

Các thông tin cần thu thập bao gồm:
- Quận/Huyện (Quan trọng nhất)
- Loại bất động sản (Chung cư, Nhà phố, Đất...)
- Diện tích (m2)
- Số phòng ngủ, số toilet
- Và các thông tin khác nếu người dùng cung cấp.

Nếu người dùng đưa link, hãy nói rằng bạn đã trích xuất thông tin từ link đó.
Nếu bạn đã có dự đoán giá, hãy thông báo cho người dùng và giải thích ngắn gọn tại sao có giá đó.
Luôn sử dụng đơn vị diện tích là m2 và tiền tệ là VNĐ (Ví dụ: 5 tỷ, 5.5 tỷ).
"""

def extract_info(state: GraphState) -> Dict[str, Any]:
    """
    Node trích xuất thông tin từ tin nhắn người dùng hoặc URL.
    """
    messages = state['messages']
    last_message = messages[-1]
    current_features = state.get('features', PropertyFeatures())
    
    # Check for URL
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, last_message.content)
    
    extracted_features = current_features.copy()
    
    if urls:
        # URL Mode
        url = urls[0]
        scraped_features = fetch_property_details(url)
        # Merge scraped features
        extracted_features_dict = extracted_features.dict(exclude_none=True)
        scraped_features_dict = scraped_features.dict(exclude_none=True)
        extracted_features_dict.update(scraped_features_dict)
        extracted_features = PropertyFeatures(**extracted_features_dict)
        return {"features": extracted_features, "user_input_url": url}
    else:
        # Text Mode - Use LLM to extract
        # Using structured output extraction
        extraction_llm = llm.with_structured_output(PropertyFeatures)
        
        # Construct prompt for extraction
        extraction_prompt = f"""
        Trích xuất thông tin bất động sản từ tin nhắn sau của người dùng vào JSON.
        Nếu không có thông tin, hãy để trống.
        Tin nhắn: {last_message.content}
        """
        
        try:
            result = extraction_llm.invoke(extraction_prompt)
            # Merge with existing
            current_dict = current_features.dict(exclude_none=True)
            new_dict = result.dict(exclude_none=True)
            current_dict.update(new_dict)
            extracted_features = PropertyFeatures(**current_dict)
        except Exception as e:
            print(f"Extraction error: {e}")
            
        return {"features": extracted_features}

def predict_price(state: GraphState) -> Dict[str, Any]:
    """
    Node dự đoán giá nếu đủ thông tin quan trọng.
    """
    features = state.get('features', PropertyFeatures())
    
    # Basic check: needs at least area and size (or other dims) to predict
    if features.area_name and (features.size or (features.width and features.length) or features.living_size):
        predictor = PricePredictor()
        price = predictor.predict(features)
        return {"prediction_result": price}
    
    return {"prediction_result": None}

def chatbot(state: GraphState) -> Dict[str, Any]:
    """
    Node sinh câu trả lời cho người dùng.
    """
    messages = state['messages']
    features = state.get('features')
    prediction = state.get('prediction_result')
    
    # Add context about current state to the system prompt (simulated)
    # Since we can't easily modify the initial system message in the list cleanly without duplication,
    # we'll append a system instruction as the second to last message or use a prompt template.
    
    status_msg = f"Hiện tại tôi đã có các thông tin sau: {features.dict(exclude_none=True)}"
    if prediction:
        status_msg += f"\nĐã có dự đoán giá: {prediction:,.0f} VNĐ."
    else:
        status_msg += "\nChưa đủ thông tin để dự đoán giá (Cần ít nhất Quận/Huyện và Diện tích)."
        
    generation_prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=status_msg),
    ] + messages
    
    response = llm.invoke(generation_prompt)
    
    return {"messages": [response]}
