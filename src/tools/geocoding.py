"""
Geocoding tool for converting addresses to latitude/longitude coordinates.
Uses Google Maps Geocoding API.
"""

import os
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
import googlemaps
from dotenv import load_dotenv

load_dotenv()


class GeocodingResult(BaseModel):
    """Result from geocoding an address."""
    latitude: Optional[float] = Field(None, description="Vĩ độ")
    longitude: Optional[float] = Field(None, description="Kinh độ")
    formatted_address: Optional[str] = Field(None, description="Địa chỉ đầy đủ")
    success: bool = Field(default=False, description="Geocoding thành công hay không")
    error: Optional[str] = Field(None, description="Thông báo lỗi nếu có")


def _get_maps_client() -> Optional[googlemaps.Client]:
    """Get Google Maps client instance."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    return googlemaps.Client(key=api_key)


@tool
def get_coordinates(address: str) -> dict:
    """Tìm tọa độ (kinh độ, vĩ độ) của một địa chỉ bất động sản tại Việt Nam.

    Sử dụng tool này khi cần xác định vị trí chính xác của bất động sản dựa trên
    địa chỉ, tên đường, phường, quận được người dùng cung cấp.

    Args:
        address: Địa chỉ cần tìm tọa độ. Ví dụ: "123 Nguyễn Huệ, Quận 1, TP.HCM"
                 hoặc "Bitexco Tower, Quận 1, Hồ Chí Minh"

    Returns:
        dict với các trường:
        - latitude: Vĩ độ (float)
        - longitude: Kinh độ (float)
        - formatted_address: Địa chỉ đầy đủ từ Google Maps
        - success: True nếu tìm được tọa độ
        - error: Thông báo lỗi nếu có
    """
    client = _get_maps_client()

    if not client:
        return {
            "latitude": None,
            "longitude": None,
            "formatted_address": None,
            "success": False,
            "error": "GOOGLE_MAPS_API_KEY không được cấu hình"
        }

    try:
        # Add Vietnam context for better results
        search_address = address
        if "việt nam" not in address.lower() and "vietnam" not in address.lower():
            search_address = f"{address}, Việt Nam"

        # Call Google Maps Geocoding API
        geocode_result = client.geocode(
            search_address,
            language="vi",  # Vietnamese language for results
            region="vn"     # Bias results to Vietnam
        )

        if not geocode_result:
            return {
                "latitude": None,
                "longitude": None,
                "formatted_address": None,
                "success": False,
                "error": f"Không tìm thấy tọa độ cho địa chỉ: {address}"
            }

        # Extract location from first result
        location = geocode_result[0]["geometry"]["location"]
        formatted_address = geocode_result[0].get("formatted_address", "")

        return {
            "latitude": location["lat"],
            "longitude": location["lng"],
            "formatted_address": formatted_address,
            "success": True,
            "error": None
        }

    except googlemaps.exceptions.ApiError as e:
        return {
            "latitude": None,
            "longitude": None,
            "formatted_address": None,
            "success": False,
            "error": f"Lỗi Google Maps API: {str(e)}"
        }
    except Exception as e:
        return {
            "latitude": None,
            "longitude": None,
            "formatted_address": None,
            "success": False,
            "error": f"Lỗi không xác định: {str(e)}"
        }
