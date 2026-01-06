from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class PropertyFeatures(BaseModel):
    """
    Mô hình dữ liệu cho các đặc điểm bất động sản.
    Tất cả các trường đều là tùy chọn để cho phép thu thập dữ liệu từng bước.
    """
    
    # Địa điểm
    area_name: Optional[str] = Field(None, description="Tên quận/huyện (ví dụ: Quận 1, Tp Thủ Đức)")
    longitude: Optional[float] = Field(None, description="Kinh độ")
    latitude: Optional[float] = Field(None, description="Vĩ độ")
    is_main_street: Optional[bool] = Field(None, description="Là nhà mặt phố/mặt tiền hay không")

    # Loại bất động sản
    category_name: Optional[str] = Field(None, description="Danh mục (ví dụ: Căn hộ/Chung cư, Nhà ở, Đất)")
    house_type_name: Optional[str] = Field(None, description="Loại nhà (ví dụ: Nhà biệt thự, Nhà phố liền kề)")
    apartment_type_name: Optional[str] = Field(None, description="Loại chung cư (ví dụ: Chung cư, Penthouse)")
    commercial_type_name: Optional[str] = Field(None, description="Loại văn phòng/thương mại")
    land_type_name: Optional[str] = Field(None, description="Loại đất")
    
    # Kích thước và Diện tích (m2)
    size: Optional[float] = Field(None, description="Diện tích đất/sử dụng (m2)")
    living_size: Optional[float] = Field(None, description="Diện tích sử dụng thực tế (m2)")
    width: Optional[float] = Field(None, description="Chiều ngang (m)")
    length: Optional[float] = Field(None, description="Chiều dài (m)")
    
    # Cấu trúc
    floors: Optional[int] = Field(None, description="Tổng số tầng")
    floornumber: Optional[int] = Field(None, description="Tầng số bao nhiêu (nếu là chung cư)")
    rooms_count: Optional[int] = Field(None, description="Số phòng ngủ")
    toilets_count: Optional[int] = Field(None, description="Số phòng vệ sinh")
    
    # Đặc điểm khác
    direction_name: Optional[str] = Field(None, description="Hướng nhà (Đông, Tây, Nam, Bắc...)")
    balconydirection_name: Optional[str] = Field(None, description="Hướng ban công")
    furnishing_sell_status: Optional[str] = Field(None, description="Tình trạng nội thất (Nội thất đầy đủ, Thô...)")
    property_legal_document_status: Optional[str] = Field(None, description="Tình trạng pháp lý (Sổ hồng, Hợp đồng...)")
    property_status_name: Optional[str] = Field(None, description="Tình trạng bàn giao dự án")

    class Config:
        json_schema_extra = {
            "example": {
                "area_name": "Quận 1",
                "category_name": "Nhà ở",
                "size": 100.5,
                "rooms_count": 3,
                "is_main_street": True
            }
        }
