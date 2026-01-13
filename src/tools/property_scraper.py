"""
Firecrawl-based property scraper tool for extracting real estate information from Vietnamese property listing URLs.
Integrates with LangChain as a tool for the chatbot.
"""
import os
import re
import requests
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Firecrawl API endpoint
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1/scrape"


class ScrapedPropertyInfo(BaseModel):
    """Schema for extracted property information from listing pages."""
    title: Optional[str] = Field(None, description="Tiêu đề tin đăng")
    actual_price: Optional[float] = Field(None, description="Giá bán (đơn vị: VNĐ, chuyển đổi từ tỷ/triệu sang số)")
    price_text: Optional[str] = Field(None, description="Giá bán gốc như hiển thị trên trang (ví dụ: 5.2 tỷ, 15 triệu/m²)")
    area_name: Optional[str] = Field(None, description="Quận/Huyện (ví dụ: Quận 1, Thành phố Thủ Đức)")
    address: Optional[str] = Field(None, description="Địa chỉ đầy đủ")
    size: Optional[float] = Field(None, description="Diện tích (m²)")
    rooms_count: Optional[str] = Field(None, description="Số phòng ngủ")
    toilets_count: Optional[str] = Field(None, description="Số phòng vệ sinh")
    floors: Optional[int] = Field(None, description="Số tầng")
    direction_name: Optional[str] = Field(None, description="Hướng nhà")
    width: Optional[float] = Field(None, description="Chiều ngang mặt tiền (m)")
    length: Optional[float] = Field(None, description="Chiều dài (m)")
    property_legal_document_status: Optional[str] = Field(None, description="Tình trạng pháp lý (sổ hồng, sổ đỏ, etc.)")
    furnishing_sell_status: Optional[str] = Field(None, description="Tình trạng nội thất")
    category_name: Optional[str] = Field(None, description="Loại bất động sản (Nhà ở, Căn hộ, Đất, etc.)")


def parse_vietnamese_price(price_text: str) -> Optional[float]:
    """
    Parse Vietnamese price format to VND.
    Examples:
        - "5.2 tỷ" -> 5_200_000_000
        - "15 triệu/m²" -> 15_000_000 (per m², not total)
        - "850 triệu" -> 850_000_000
        - "1,5 tỷ" -> 1_500_000_000
    """
    if not price_text:
        return None

    # Normalize text
    text = price_text.lower().strip()
    text = text.replace(",", ".").replace(" ", "")

    # Skip price per m² (these are not actual prices)
    if "/m" in text or "m²" in text or "m2" in text:
        return None

    # Extract number
    number_match = re.search(r"([\d.]+)", text)
    if not number_match:
        return None

    try:
        number = float(number_match.group(1))
    except ValueError:
        return None

    # Convert based on unit
    if "tỷ" in text or "ty" in text:
        return number * 1_000_000_000
    elif "triệu" in text or "trieu" in text:
        return number * 1_000_000
    elif "nghìn" in text or "nghin" in text or "k" in text:
        return number * 1_000
    else:
        # Assume raw VND if no unit
        return number if number > 1_000_000 else None


def scrape_property_with_firecrawl(url: str) -> Dict[str, Any]:
    """
    Scrape property information from a URL using Firecrawl API directly.

    Args:
        url: The property listing URL to scrape

    Returns:
        Dictionary with extracted property information
    """
    api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        return {
            "success": False,
            "error": "FIRECRAWL_API_KEY không được cấu hình. Vui lòng thêm API key vào file .env"
        }

    try:
        # Build schema from Pydantic model
        schema = ScrapedPropertyInfo.model_json_schema()

        # Extraction prompt for LLM
        extraction_prompt = """
        Trích xuất thông tin bất động sản từ trang tin đăng này.
        Lưu ý:
        - Chuyển đổi giá từ "tỷ" sang số (1 tỷ = 1,000,000,000 VNĐ)
        - Chuyển đổi giá từ "triệu" sang số (1 triệu = 1,000,000 VNĐ)
        - Bỏ qua giá theo m² (như "15 triệu/m²"), chỉ lấy tổng giá
        - Diện tích lấy số (ví dụ: 100 từ "100 m²")
        - Quận/Huyện: lấy chính xác tên (Quận 1, Thành phố Thủ Đức, etc.)

        QUAN TRỌNG - Phân biệt số phòng ngủ và phòng vệ sinh:
        - "PN" hoặc "phòng ngủ" = số phòng ngủ (rooms_count). Ví dụ: "3 PN" -> rooms_count = "3"
        - "WC", "VS", "toilet", "phòng vệ sinh", "nhà vệ sinh" = số phòng vệ sinh (toilets_count). Ví dụ: "2 WC" -> toilets_count = "2"
        - KHÔNG được lấy số từ "PN" để gán cho toilets_count
        - Nếu chỉ thấy "3 PN" mà không có thông tin WC/VS riêng, thì toilets_count = null
        - Chỉ điền toilets_count khi có thông tin rõ ràng về WC/VS/phòng vệ sinh
        """

        # Prepare request payload
        payload = {
            "url": url,
            "formats": ["extract"],
            "extract": {
                "prompt": extraction_prompt,
                "schema": schema
            }
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Make API request
        response = requests.post(FIRECRAWL_API_URL, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Firecrawl API error: {response.status_code} - {response.text}",
                "url": url
            }

        result = response.json()

        # Check for successful response
        if result.get("success") and result.get("data"):
            data = result["data"]
            extracted = data.get("extract", {})

            # Post-process: parse price if it wasn't converted properly
            if extracted.get("actual_price") is None and extracted.get("price_text"):
                extracted["actual_price"] = parse_vietnamese_price(extracted["price_text"])

            return {
                "success": True,
                "data": extracted,
                "url": url
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Không thể trích xuất thông tin từ trang này"),
                "url": url
            }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout khi kết nối với Firecrawl API",
            "url": url
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Lỗi kết nối: {str(e)}",
            "url": url
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Lỗi khi scrape trang: {str(e)}",
            "url": url
        }


@tool
def scrape_property_listing(url: str) -> Dict[str, Any]:
    """
    Trích xuất thông tin bất động sản từ URL tin đăng.

    Sử dụng công cụ này khi người dùng cung cấp link bất động sản từ các trang như:
    - batdongsan.com.vn
    - chotot.com
    - alonhadat.com.vn
    - mogi.vn
    - nha.chotot.com
    - homedy.com

    Công cụ sẽ trả về:
    - Giá bất động sản (actual_price)
    - Diện tích (size)
    - Địa chỉ và quận/huyện (area_name)
    - Số phòng ngủ/vệ sinh
    - Các thông tin khác từ tin đăng

    Args:
        url: Link tin đăng bất động sản cần trích xuất thông tin

    Returns:
        Dictionary chứa thông tin bất động sản đã trích xuất
    """
    # Validate URL format
    if not url or not url.startswith(("http://", "https://")):
        return {
            "success": False,
            "error": "URL không hợp lệ. Vui lòng cung cấp đường link đầy đủ (bắt đầu bằng http:// hoặc https://)"
        }

    # Check if it's a known Vietnamese real estate site
    known_domains = [
        "batdongsan.com.vn",
        "chotot.com",
        "nhatot.com",
        "nha.chotot.com",
        "alonhadat.com.vn",
        "mogi.vn",
        "homedy.com",
        "muaban.net",
        "nhadat247.com.vn"
    ]

    is_known_site = any(domain in url.lower() for domain in known_domains)

    # Proceed with scraping
    result = scrape_property_with_firecrawl(url)

    if not is_known_site and result.get("success"):
        result["warning"] = "Đây không phải trang bất động sản Việt Nam phổ biến. Kết quả có thể không chính xác."

    return result


def extract_features_from_scraped_data(scraped_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert scraped data to PropertyFeatures-compatible format.

    Args:
        scraped_data: The data returned from scrape_property_listing

    Returns:
        Dictionary with fields compatible with PropertyFeatures model
    """
    if not scraped_data.get("success") or not scraped_data.get("data"):
        return {}

    data = scraped_data["data"]
    features = {}

    # Direct mappings
    direct_fields = [
        "actual_price", "size", "width", "length", "floors",
        "area_name", "direction_name", "category_name",
        "rooms_count", "toilets_count",
        "property_legal_document_status", "furnishing_sell_status"
    ]

    for field in direct_fields:
        if data.get(field) is not None:
            features[field] = data[field]

    return features
