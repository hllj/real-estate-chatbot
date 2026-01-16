"""
Listing Search Tool for finding similar real estate listings.

This tool allows the chatbot to search for similar properties from the database
based on user criteria, with intelligent fallback when exact matches aren't found.
"""

from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.models import PropertyFeatures
from src.ml.listing_recommender import get_recommender


class ListingSearchInput(BaseModel):
    """Input schema for listing search tool."""
    area_name: Optional[str] = Field(None, description="Quận/Huyện (ví dụ: 'Quận 1', 'Quận 7', 'Thành phố Thủ Đức')")
    category_name: Optional[str] = Field(None, description="Loại BĐS (Căn hộ/Chung cư, Nhà ở, Đất, Văn phòng)")
    target_price: Optional[float] = Field(None, description="Giá mục tiêu (VNĐ) để tìm BĐS trong khoảng giá tương tự")
    size: Optional[float] = Field(None, description="Diện tích mục tiêu (m²)")
    rooms_count: Optional[str] = Field(None, description="Số phòng ngủ (1-10 hoặc 'nhiều hơn 10')")
    house_type_name: Optional[str] = Field(None, description="Loại nhà (Nhà biệt thự, Nhà phố liền kề, Nhà ngõ hẻm, Nhà mặt phố mặt tiền)")
    apartment_type_name: Optional[str] = Field(None, description="Loại căn hộ (Chung cư, Penthouse, Duplex, Officetel)")
    mode: str = Field(default="Sell", description="Chế độ tìm kiếm: 'Sell' (mua bán) hoặc 'Rent' (cho thuê)")
    max_results: int = Field(default=5, description="Số lượng kết quả tối đa (1-10)")


@tool(args_schema=ListingSearchInput)
def search_similar_listings(
    area_name: Optional[str] = None,
    category_name: Optional[str] = None,
    target_price: Optional[float] = None,
    size: Optional[float] = None,
    rooms_count: Optional[str] = None,
    house_type_name: Optional[str] = None,
    apartment_type_name: Optional[str] = None,
    mode: str = "Sell",
    max_results: int = 5
) -> dict:
    """Tìm kiếm các bất động sản tương tự trong cơ sở dữ liệu.

    Công cụ này tìm kiếm các bất động sản tương tự dựa trên tiêu chí người dùng cung cấp.
    Nếu không tìm thấy kết quả chính xác, hệ thống sẽ tự động nới lỏng tiêu chí
    để tìm các BĐS gần giống nhất.

    Sử dụng tool này khi:
    - Người dùng muốn xem các BĐS tương tự với tiêu chí họ đưa ra
    - Người dùng muốn so sánh giá với các BĐS cùng loại trên thị trường
    - Sau khi đã có dự đoán giá và muốn gợi ý các BĐS trong tầm giá
    - Người dùng hỏi "có BĐS nào tương tự không?", "giá này có hợp lý không?"

    Args:
        area_name: Quận/Huyện (ví dụ: 'Quận 1', 'Quận 7')
        category_name: Loại BĐS (Căn hộ/Chung cư, Nhà ở, Đất, Văn phòng)
        target_price: Giá mục tiêu (VNĐ) - thường là giá dự đoán hoặc giá người dùng quan tâm
        size: Diện tích mục tiêu (m²)
        rooms_count: Số phòng ngủ ('1', '2', '3', ... hoặc 'nhiều hơn 10')
        house_type_name: Loại nhà nếu là Nhà ở
        apartment_type_name: Loại căn hộ nếu là Chung cư
        mode: 'Sell' để tìm BĐS bán, 'Rent' để tìm BĐS cho thuê
        max_results: Số lượng kết quả tối đa (mặc định 5, tối đa 10)

    Returns:
        dict với:
        - success: True nếu tìm được kết quả
        - total_found: Số lượng BĐS tìm được
        - listings: Danh sách các BĐS với thông tin chi tiết
        - search_criteria: Tiêu chí tìm kiếm đã sử dụng
        - relaxation_applied: Các tiêu chí đã được nới lỏng (nếu có)
        - message: Thông báo tóm tắt bằng tiếng Việt
    """
    # Validate and cap max_results
    max_results = min(max(1, max_results), 10)

    # Create PropertyFeatures from input
    features = PropertyFeatures(
        area_name=area_name,
        category_name=category_name,
        size=size,
        rooms_count=rooms_count,
        house_type_name=house_type_name,
        apartment_type_name=apartment_type_name,
    )

    # Get recommender for the appropriate mode
    try:
        recommender = get_recommender(mode=mode)
    except Exception as e:
        return {
            "success": False,
            "error": f"Không thể khởi tạo hệ thống tìm kiếm: {str(e)}",
            "listings": [],
            "message": f"Lỗi hệ thống: {str(e)}"
        }

    # Perform search
    try:
        result = recommender.search_listings(
            features=features,
            predicted_price=target_price,
            max_results=max_results
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "listings": [],
            "message": f"Lỗi khi tìm kiếm: {str(e)}"
        }


def search_listings_from_features(
    features: PropertyFeatures,
    predicted_price: Optional[float] = None,
    mode: str = "Sell",
    max_results: int = 5
) -> dict:
    """
    Helper function to search listings directly from PropertyFeatures object.
    Used internally by the chatbot graph.

    Args:
        features: PropertyFeatures object with search criteria
        predicted_price: Optional predicted price for price range filtering
        mode: "Sell" or "Rent"
        max_results: Maximum number of results

    Returns:
        Search results dictionary
    """
    try:
        recommender = get_recommender(mode=mode)
        return recommender.search_listings(
            features=features,
            predicted_price=predicted_price,
            max_results=max_results
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "listings": [],
            "message": f"Lỗi khi tìm kiếm: {str(e)}"
        }
