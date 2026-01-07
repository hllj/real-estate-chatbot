import os
import re
from typing import Dict, Any, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers.openai_functions import PydanticOutputFunctionsParser
from dotenv import load_dotenv

from src.graph.state import GraphState
from src.models import PropertyFeatures, VALID_VALUES
from src.ml.real_estate_predictor import PricePredictor
from src.utils.scraper import fetch_property_details
from src.tools.geocoding import get_coordinates

load_dotenv()

# Setup LLM
llm = ChatGoogleGenerativeAI(model=os.environ["GEMINI_MODEL"], temperature=0, convert_system_message_to_human=True)

# Setup LLM with tools for geocoding
tools = [get_coordinates]
llm_with_tools = llm.bind_tools(tools)

# Format valid values for prompts
def format_valid_values_for_prompt() -> str:
    """Format VALID_VALUES dictionary into a readable string for LLM prompts."""
    lines = []
    for field, values in VALID_VALUES.items():
        # Filter out "Không có thông tin" for cleaner display
        display_values = [v for v in values if v != "Không có thông tin"]
        lines.append(f"- {field}: {', '.join(display_values)}")
    return "\n".join(lines)

VALID_VALUES_PROMPT = format_valid_values_for_prompt()

# Vietnamese System Prompt with valid values guidance
SYSTEM_PROMPT = f"""Bạn là một trợ lý thông minh chuyên về bất động sản tại Việt Nam.
Nhiệm vụ của bạn là thu thập thông tin từ người dùng để dự đoán giá nhà.
Hãy giao tiếp bằng tiếng Việt một cách tự nhiên, chuyên nghiệp và thân thiện.

Mục tiêu của bạn là thu thập đầy đủ các thông tin sau để có dự đoán chính xác nhất:

1.  **Vị trí (Quan trọng nhất):**
    *   Quận/Huyện (`area_name`) - Chỉ chấp nhận các quận/huyện tại TP.HCM:
        {', '.join(VALID_VALUES['area_name'])}. Những thông tin này được sử dụng để tính longitude và latitude nội bộ. Nếu được hãy hỏi người dùng về tên đường, ghi nhận thêm nếu có tên đường, phường nếu có. Sử dụng tool get_coordinates.
    *   Đặc điểm vị trí: Mặt tiền đường lớn (`is_main_street`) hay hẻm?

2.  **Loại Bất Động Sản:**
    *   Danh mục chính (`category_name`): {', '.join(VALID_VALUES['category_name'])}
    *   Chi tiết loại hình:
        *   Nếu là Nhà ở (`house_type_name`): {', '.join([v for v in VALID_VALUES['house_type_name'] if v != 'Không có thông tin'])}
        *   Nếu là Chung cư (`apartment_type_name`): {', '.join([v for v in VALID_VALUES['apartment_type_name'] if v != 'Không có thông tin'])}
        *   Nếu là Đất (`land_type_name`): {', '.join([v for v in VALID_VALUES['land_type_name'] if v != 'Không có thông tin'])}
        *   Nếu là Văn phòng/TM (`commercial_type_name`): {', '.join([v for v in VALID_VALUES['commercial_type_name'] if v != 'Không có thông tin'])}

3.  **Kích thước & Diện tích:**
    *   Diện tích đất/sử dụng (`size`) - Đơn vị: m2. Hãy hỏi trong mọi trường hợp.
    *   Diện tích sử dụng thực tế (`living_size`) - Đơn vị: m2. Hãy hỏi trong mọi trường hợp.
    *   Kích thước: Chiều ngang (`width`) x Chiều dài (`length`). Hãy hỏi trong mọi trường hợp.

4.  **Cấu trúc & Tiện ích:**
    *   Số tầng (`floors`). Chỉ hỏi trong trường hợp chung cư.
    *   Tầng số mấy (`floornumber`) - Nếu là tìm căn hộ / chung cư.
    *   Số phòng ngủ (`rooms_count`): 1-10 hoặc "nhiều hơn 10".
    *   Số toilet (`toilets_count`): 1-6 hoặc "Nhiều hơn 6".
    *   Hướng nhà (`direction_name`): {', '.join([v for v in VALID_VALUES['direction_name'] if v != 'Không có thông tin'])}
    *   Hướng ban công (`balconydirection_name`): {', '.join([v for v in VALID_VALUES['balconydirection_name'] if v != 'Không có thông tin'])}
    *   Nội thất (`furnishing_sell_status`): {', '.join([v for v in VALID_VALUES['furnishing_sell_status'] if v != 'Không có thông tin'])}

5.  **Pháp lý & Tình trạng:**
    *   Giấy tờ pháp lý (`property_legal_document_status`): {', '.join([v for v in VALID_VALUES['property_legal_document_status'] if v != 'Không có thông tin'])}
    *   Tình trạng bàn giao (`property_status_name`): {', '.join([v for v in VALID_VALUES['property_status_name'] if v != 'Không có thông tin'])}

Lưu ý:
*   Nếu người dùng đưa link, hãy nói rằng bạn đã trích xuất thông tin từ link đó.
*   Nếu bạn đã có dự đoán giá, hãy thông báo cho người dùng và giải thích ngắn gọn tại sao có giá đó.
*   Luôn sử dụng đơn vị diện tích là m2 và tiền tệ là VNĐ (Ví dụ: 5 tỷ, 5.5 tỷ).
*   Đừng hỏi dồn dập tất cả cùng lúc. Hãy hỏi tự nhiên, ưu tiên Vị trí và Loại bất động sản trước.
*   Khi hỏi người dùng về thông tin, hãy gợi ý các lựa chọn hợp lệ để họ dễ trả lời.

**QUAN TRỌNG - Xử lý thông tin người dùng không biết:**
*   Nếu người dùng nói họ "không biết", "không rõ", "chưa biết", "không nhớ" về một trường nào đó, hãy GHI NHẬN và KHÔNG hỏi lại về trường đó nữa.
*   Chấp nhận rằng một số thông tin có thể không có và tiếp tục với các thông tin khác.
*   Chỉ hỏi lại về một trường đã đánh dấu "không biết" nếu người dùng CHỦ ĐỘNG cung cấp thông tin mới.
*   Ví dụ: Nếu người dùng nói "không biết hướng nhà", KHÔNG hỏi lại "Hướng nhà là gì?" trong các câu hỏi tiếp theo.
"""

# Extraction prompt with strict valid values
EXTRACTION_PROMPT_TEMPLATE = """Trích xuất thông tin bất động sản từ tin nhắn sau của người dùng.

**QUAN TRỌNG:** Chỉ sử dụng các giá trị hợp lệ sau cho các trường categorical:

{valid_values}

**Quy tắc:**
1. Nếu người dùng nói "Q1", "Quận 1", "quận 1" → area_name = "Quận 1"
2. Nếu người dùng nói "Thủ Đức", "TP Thủ Đức" → area_name = "Thành phố Thủ Đức"
3. Nếu người dùng nói "nhà phố", "nhà mặt tiền" → house_type_name = "Nhà mặt phố, mặt tiền"
4. Nếu người dùng nói "biệt thự" → house_type_name = "Nhà biệt thự"
5. Nếu người dùng nói "chung cư", "căn hộ" → category_name = "Căn hộ/Chung cư"
6. Nếu người dùng nói số phòng ngủ là số (vd: 3) → rooms_count = "3" (string)
7. Nếu người dùng nói hướng "Đông Nam", "đông nam", "ĐN" → direction_name = "Đông Nam"
8. Nếu không có thông tin cho một trường → để trống (null), KHÔNG dùng "Không có thông tin"
9. Với các trường số (size, width, length, floors, floornumber) → dùng số, không dùng string

**Xử lý thông tin không rõ:**
- Nếu người dùng NÓI RÕ họ "không biết", "không rõ", "chưa biết", "chưa có thông tin", "không có", "không nhớ", "chịu", "không chắc" về một trường cụ thể, hãy đặt giá trị đó là "UNKNOWN_FIELD"
- Ví dụ: "Tôi không biết hướng nhà" → direction_name = "UNKNOWN_FIELD"
- Ví dụ: "Không rõ số phòng ngủ" → rooms_count = "UNKNOWN_FIELD"
- Ví dụ: "Chịu, không nhớ diện tích" → size = "UNKNOWN_FIELD" (đây là trường số nhưng vẫn dùng marker)
- CHỈ đánh dấu UNKNOWN_FIELD khi người dùng MINH BẠCH nói họ không biết
- KHÔNG đánh dấu UNKNOWN_FIELD cho thông tin người dùng không đề cập đến

**Tin nhắn người dùng:**
{user_message}

Trích xuất thông tin vào JSON schema đã định nghĩa.
"""


def extract_info(state: GraphState) -> Dict[str, Any]:
    """
    Node trích xuất thông tin từ tin nhắn người dùng hoặc URL.
    """
    messages = state['messages']
    last_message = messages[-1]
    current_features = state.get('features', PropertyFeatures())
    current_unknown_fields = state.get('unknown_fields', [])

    # Check for URL
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, last_message.content)

    extracted_features = current_features.model_copy()
    unknown_fields = list(current_unknown_fields)

    if urls:
        # URL Mode
        url = urls[0]
        scraped_features = fetch_property_details(url)
        # Merge scraped features
        extracted_features_dict = extracted_features.model_dump(exclude_none=True)
        scraped_features_dict = scraped_features.model_dump(exclude_none=True)
        extracted_features_dict.update(scraped_features_dict)
        # Validate and normalize scraped features
        extracted_features, unknown_fields = validate_and_normalize_features(
            extracted_features_dict, current_unknown_fields
        )
        return {"features": extracted_features, "user_input_url": url, "unknown_fields": unknown_fields}
    else:
        # Text Mode - Use LLM to extract with strict valid values
        extraction_llm = llm.with_structured_output(PropertyFeatures)

        # Construct prompt for extraction with valid values
        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            valid_values=VALID_VALUES_PROMPT,
            user_message=last_message.content
        )

        try:
            result = extraction_llm.invoke(extraction_prompt)
            # Merge with existing
            current_dict = current_features.model_dump(exclude_none=True)
            new_dict = result.model_dump(exclude_none=True)
            current_dict.update(new_dict)
            # Validate and normalize
            extracted_features, unknown_fields = validate_and_normalize_features(
                current_dict, current_unknown_fields
            )
        except Exception as e:
            print(f"Extraction error: {e}")

        return {"features": extracted_features, "unknown_fields": unknown_fields}


def validate_and_normalize_features(
    features_dict: Dict[str, Any],
    existing_unknown_fields: List[str] = None
) -> Tuple[PropertyFeatures, List[str]]:
    """
    Validate and normalize extracted features to match valid values.
    Uses fuzzy matching for common variations.

    Returns:
        Tuple of (validated PropertyFeatures, list of unknown field names)
    """
    from difflib import get_close_matches

    # Mapping of common abbreviations and variations
    AREA_NAME_ALIASES = {
        "q1": "Quận 1", "q2": "Quận 2", "q3": "Quận 3", "q4": "Quận 4",
        "q5": "Quận 5", "q6": "Quận 6", "q7": "Quận 7", "q8": "Quận 8",
        "q9": "Thành phố Thủ Đức", "q10": "Quận 10", "q11": "Quận 11", "q12": "Quận 12",
        "quận 9": "Thành phố Thủ Đức", "quận 2": "Thành phố Thủ Đức",
        "thủ đức": "Thành phố Thủ Đức", "tp thủ đức": "Thành phố Thủ Đức",
        "thành phố thủ đức": "Thành phố Thủ Đức",
        "bình thạnh": "Quận Bình Thạnh", "gò vấp": "Quận Gò Vấp",
        "tân bình": "Quận Tân Bình", "tân phú": "Quận Tân Phú",
        "phú nhuận": "Quận Phú Nhuận", "bình tân": "Quận Bình Tân",
        "củ chi": "Huyện Củ Chi", "hóc môn": "Huyện Hóc Môn",
        "bình chánh": "Huyện Bình Chánh", "nhà bè": "Huyện Nhà Bè",
        "cần giờ": "Huyện Cần Giờ",
    }

    DIRECTION_ALIASES = {
        "đông": "Đông", "tây": "Tây", "nam": "Nam", "bắc": "Bắc",
        "đông nam": "Đông Nam", "đông bắc": "Đông Bắc",
        "tây nam": "Tây Nam", "tây bắc": "Tây Bắc",
        "đn": "Đông Nam", "đb": "Đông Bắc", "tn": "Tây Nam", "tb": "Tây Bắc",
    }

    normalized = features_dict.copy()
    unknown_fields = list(existing_unknown_fields) if existing_unknown_fields else []

    # Detect fields marked as UNKNOWN_FIELD by LLM
    UNKNOWN_MARKER = "UNKNOWN_FIELD"
    for field, value in list(normalized.items()):
        if value == UNKNOWN_MARKER or (isinstance(value, str) and UNKNOWN_MARKER in value):
            if field not in unknown_fields:
                unknown_fields.append(field)
            normalized[field] = None
            print(f"Field '{field}' marked as unknown by user")

    # Normalize area_name
    if "area_name" in normalized and normalized["area_name"]:
        area_lower = normalized["area_name"].lower().strip()
        if area_lower in AREA_NAME_ALIASES:
            normalized["area_name"] = AREA_NAME_ALIASES[area_lower]
        elif normalized["area_name"] not in VALID_VALUES["area_name"]:
            # Try fuzzy matching with strict cutoff
            matches = get_close_matches(
                normalized["area_name"],
                VALID_VALUES["area_name"],
                n=1,
                cutoff=0.8
            )
            if matches:
                print(f"Normalized area_name: '{normalized['area_name']}' -> '{matches[0]}'")
                normalized["area_name"] = matches[0]
            else:
                print(f"Warning: Invalid area_name '{normalized['area_name']}', setting to None")
                normalized["area_name"] = None

    # Normalize direction_name
    if "direction_name" in normalized and normalized["direction_name"]:
        dir_lower = normalized["direction_name"].lower().strip()
        if dir_lower in DIRECTION_ALIASES:
            normalized["direction_name"] = DIRECTION_ALIASES[dir_lower]
        elif normalized["direction_name"] not in VALID_VALUES["direction_name"]:
            matches = get_close_matches(
                normalized["direction_name"],
                VALID_VALUES["direction_name"],
                n=1,
                cutoff=0.8
            )
            if matches:
                normalized["direction_name"] = matches[0]
            else:
                normalized["direction_name"] = None

    # Normalize balconydirection_name
    if "balconydirection_name" in normalized and normalized["balconydirection_name"]:
        dir_lower = normalized["balconydirection_name"].lower().strip()
        if dir_lower in DIRECTION_ALIASES:
            normalized["balconydirection_name"] = DIRECTION_ALIASES[dir_lower]
        elif normalized["balconydirection_name"] not in VALID_VALUES["balconydirection_name"]:
            matches = get_close_matches(
                normalized["balconydirection_name"],
                VALID_VALUES["balconydirection_name"],
                n=1,
                cutoff=0.8
            )
            if matches:
                normalized["balconydirection_name"] = matches[0]
            else:
                normalized["balconydirection_name"] = None

    # Normalize rooms_count (convert int to string if needed)
    if "rooms_count" in normalized and normalized["rooms_count"] is not None:
        val = normalized["rooms_count"]
        if isinstance(val, int):
            if val > 10:
                normalized["rooms_count"] = "nhiều hơn 10"
            else:
                normalized["rooms_count"] = str(val)
        elif isinstance(val, str) and val not in VALID_VALUES["rooms_count"]:
            # Try to convert to valid format
            try:
                num = int(val)
                if num > 10:
                    normalized["rooms_count"] = "nhiều hơn 10"
                else:
                    normalized["rooms_count"] = str(num)
            except ValueError:
                normalized["rooms_count"] = None

    # Normalize toilets_count (convert int to string if needed)
    if "toilets_count" in normalized and normalized["toilets_count"] is not None:
        val = normalized["toilets_count"]
        if isinstance(val, int):
            if val > 6:
                normalized["toilets_count"] = "Nhiều hơn 6"
            else:
                normalized["toilets_count"] = str(val)
        elif isinstance(val, str) and val not in VALID_VALUES["toilets_count"]:
            try:
                num = int(val)
                if num > 6:
                    normalized["toilets_count"] = "Nhiều hơn 6"
                else:
                    normalized["toilets_count"] = str(num)
            except ValueError:
                normalized["toilets_count"] = None

    # Validate other categorical fields
    categorical_fields = [
        "category_name", "apartment_type_name", "house_type_name",
        "commercial_type_name", "land_type_name", "furnishing_sell_status",
        "property_legal_document_status", "property_status_name"
    ]

    for field in categorical_fields:
        if field in normalized and normalized[field]:
            if normalized[field] == "Không có thông tin":
                normalized[field] = None
            elif normalized[field] not in VALID_VALUES.get(field, []):
                matches = get_close_matches(
                    normalized[field],
                    VALID_VALUES.get(field, []),
                    n=1,
                    cutoff=0.8
                )
                if matches:
                    print(f"Normalized {field}: '{normalized[field]}' -> '{matches[0]}'")
                    normalized[field] = matches[0]
                else:
                    print(f"Warning: Invalid {field} '{normalized[field]}', setting to None")
                    normalized[field] = None

    # Remove "Không có thông tin" from any remaining fields
    for key, value in list(normalized.items()):
        if value == "Không có thông tin":
            normalized[key] = None

    # If user provides a new value for a previously unknown field, remove it from unknown_fields
    for field in list(unknown_fields):
        if field in normalized and normalized[field] is not None:
            unknown_fields.remove(field)
            print(f"Field '{field}' updated by user, removing from unknown_fields")

    try:
        return PropertyFeatures(**normalized), unknown_fields
    except Exception as e:
        print(f"Validation error: {e}")
        # Return with only valid fields
        valid_normalized = {}
        for key, value in normalized.items():
            try:
                PropertyFeatures(**{key: value})
                valid_normalized[key] = value
            except Exception:
                print(f"Skipping invalid field {key}={value}")
        return PropertyFeatures(**valid_normalized), unknown_fields


def predict_price(state: GraphState) -> Dict[str, Any]:
    """
    Node dự đoán giá nếu đủ thông tin quan trọng.
    Sử dụng predict_with_confidence() để có thêm khoảng tin cậy 95%.
    """
    features = state.get('features', PropertyFeatures())

    # Basic check: needs at least area and size (or other dims) to predict
    if features.area_name and (features.size or (features.width and features.length) or features.living_size):
        predictor = PricePredictor()
        # Sử dụng predict_with_confidence để có thêm khoảng tin cậy
        prediction_result = predictor.predict_with_confidence(features)
        return {"prediction_result": prediction_result}

    return {"prediction_result": None}


def chatbot(state: GraphState) -> Dict[str, Any]:
    """
    Node sinh câu trả lời cho người dùng.
    Hỗ trợ tool calling để lấy tọa độ từ địa chỉ.
    """
    messages = state['messages']
    features = state.get('features', PropertyFeatures())
    prediction = state.get('prediction_result')
    unknown_fields = state.get('unknown_fields', [])

    # Add context about current state to the system prompt
    status_msg = f"Hiện tại tôi đã có các thông tin sau: {features.model_dump(exclude_none=True)}"
    if prediction and isinstance(prediction, dict) and prediction.get("predicted_price"):
        price = prediction["predicted_price"]
        status_msg += f"\nĐã có dự đoán giá: {price:,.0f} VNĐ."
        
        # Show confidence interval if available
        confidence = prediction.get("confidence_interval_95")
        if confidence and len(confidence) == 2:
            lower, upper = confidence
            status_msg += f"\nKhoảng tin cậy 95%: {lower:,.0f} - {upper:,.0f} VNĐ."
        
        # Indicate if using fallback model
        if prediction.get("is_fallback"):
            status_msg += "\n(Lưu ý: Sử dụng mô hình dự báo thay thế do mô hình chính không khả dụng.)"
    elif prediction and isinstance(prediction, (int, float)):
        # Backward compatibility for old format
        status_msg += f"\nĐã có dự đoán giá: {prediction:,.0f} VNĐ."
    else:
        status_msg += "\nChưa đủ thông tin để dự đoán giá (Cần ít nhất Quận/Huyện và Diện tích)."

    # Add information about unknown fields
    if unknown_fields:
        field_names_vn = {
            "area_name": "Quận/Huyện",
            "direction_name": "Hướng nhà",
            "balconydirection_name": "Hướng ban công",
            "rooms_count": "Số phòng ngủ",
            "toilets_count": "Số toilet",
            "size": "Diện tích",
            "living_size": "Diện tích sử dụng",
            "width": "Chiều ngang",
            "length": "Chiều dài",
            "floors": "Số tầng",
            "floornumber": "Tầng số",
            "category_name": "Loại BĐS",
            "house_type_name": "Loại nhà",
            "apartment_type_name": "Loại căn hộ",
            "land_type_name": "Loại đất",
            "commercial_type_name": "Loại thương mại",
            "furnishing_sell_status": "Nội thất",
            "property_legal_document_status": "Pháp lý",
            "property_status_name": "Tình trạng bàn giao",
            "is_main_street": "Mặt tiền",
        }
        unknown_names = [field_names_vn.get(f, f) for f in unknown_fields]
        status_msg += f"\n\n**CÁC THÔNG TIN NGƯỜI DÙNG ĐÃ NÓI KHÔNG BIẾT (KHÔNG HỎI LẠI):** {', '.join(unknown_names)}"

    # Add tool usage instruction
    tool_instruction = """
**CÔNG CỤ TÌM TỌA ĐỘ:**
Bạn có thể sử dụng công cụ `get_coordinates` để tìm tọa độ (kinh độ, vĩ độ) từ địa chỉ.
Sử dụng công cụ này khi:
- Người dùng cung cấp địa chỉ cụ thể (số nhà, tên đường, phường)
- Cần xác định vị trí chính xác của bất động sản
- Chưa có thông tin longitude/latitude trong features

Ví dụ địa chỉ: "123 Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM"
"""

    generation_prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=status_msg),
        SystemMessage(content=tool_instruction),
    ] + messages

    # Use LLM with tools
    response = llm_with_tools.invoke(generation_prompt)

    # Handle tool calls if any
    updated_features = features.model_copy()
    if response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "get_coordinates":
                # Execute the geocoding tool
                result = get_coordinates.invoke(tool_call["args"])

                # Update features with coordinates if successful
                if result.get("success"):
                    updated_features.latitude = result.get("latitude")
                    updated_features.longitude = result.get("longitude")

                # Create tool message with result
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

        # If there were tool calls, get final response with tool results
        if tool_messages:
            final_prompt = generation_prompt + [response] + tool_messages
            final_response = llm_with_tools.invoke(final_prompt)
            return {
                "messages": [final_response],
                "features": updated_features
            }

    return {"messages": [response], "features": updated_features}
