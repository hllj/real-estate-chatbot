from typing import Optional, Literal
from pydantic import BaseModel, Field

# Valid values for categorical features based on u_features.md and s_features.md

# Mode type for Sell vs Rent
ModeType = Literal["Sell", "Rent"]

AreaNameType = Literal[
    "Quận 1",
    "Quận 3",
    "Quận 4",
    "Quận 5",
    "Quận 6",
    "Quận 7",
    "Quận 8",
    "Quận 10",
    "Quận 11",
    "Quận 12",
    "Thành phố Thủ Đức",
    "Quận Tân Phú",
    "Huyện Củ Chi",
    "Quận Bình Tân",
    "Huyện Bình Chánh",
    "Huyện Hóc Môn",
    "Quận Gò Vấp",
    "Quận Bình Thạnh",
    "Huyện Cần Giờ",
    "Quận Tân Bình",
    "Huyện Nhà Bè",
    "Quận Phú Nhuận",
]

CategoryNameType = Literal[
    "Căn hộ/Chung cư",
    "Đất",
    "Văn phòng, Mặt bằng kinh doanh",
    "Nhà ở",
]

ApartmentTypeNameType = Literal[
    "Chung cư",
    "Duplex",
    "Căn hộ dịch vụ, mini",
    "Officetel",
    "Tập thể, cư xá",
    "Penthouse",
    "Không có thông tin",
]

PropertyLegalDocumentStatusType = Literal[
    "Hợp đồng mua bán",
    "Sổ hồng riêng",
    "Đã có sổ",
    "Đang chờ sổ",
    "Hợp đồng đặt cọc",
    "Giấy tờ khác",
    "Không có thông tin",
]

RoomsCountType = Literal[
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "nhiều hơn 10",
    "Không có thông tin",
]

ToiletsCountType = Literal[
    "1", "2", "3", "4", "5", "6",
    "Nhiều hơn 6",
    "Không có thông tin",
]

FurnishingSellStatusType = Literal[
    "Nội thất đầy đủ",
    "Nội thất cao cấp",
    "Hoàn thiện cơ bản",
    "Bàn giao thô",
    "Không có thông tin",
]

FurnishingRentStatusType = Literal[
    "Nội thất đầy đủ",
    "Nội thất cao cấp",
    "Hoàn thiện cơ bản",
    "Bàn giao thô",
    "Không có thông tin",
]

BalconyDirectionNameType = Literal[
    "Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc",
    "Không có thông tin",
]

DirectionNameType = Literal[
    "Đông", "Tây", "Nam", "Bắc", "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc",
    "Không có thông tin",
]

HouseTypeNameType = Literal[
    "Nhà ngõ, hẻm",
    "Nhà mặt phố, mặt tiền",
    "Nhà phố liền kề",
    "Nhà biệt thự",
    "Không có thông tin",
]

CommercialTypeNameType = Literal[
    "Mặt bằng kinh doanh",
    "Shophouse",
    "Văn phòng",
    "Officetel",
    "Không có thông tin",
]

LandTypeNameType = Literal[
    "Đất thổ cư",
    "Đất nền dự án",
    "Đất nông nghiệp",
    "Đất công nghiệp",
    "Không có thông tin",
]

PropertyStatusNameType = Literal[
    "Chưa bàn giao",
    "Đã bàn giao",
    "Không có thông tin",
]

# Export all valid values as lists for use in prompts and validation
VALID_VALUES = {
    "mode": list(ModeType.__args__),
    "area_name": list(AreaNameType.__args__),
    "category_name": list(CategoryNameType.__args__),
    "apartment_type_name": list(ApartmentTypeNameType.__args__),
    "property_legal_document_status": list(PropertyLegalDocumentStatusType.__args__),
    "rooms_count": list(RoomsCountType.__args__),
    "toilets_count": list(ToiletsCountType.__args__),
    "furnishing_sell_status": list(FurnishingSellStatusType.__args__),
    "furnishing_rent_status": list(FurnishingRentStatusType.__args__),
    "balconydirection_name": list(BalconyDirectionNameType.__args__),
    "direction_name": list(DirectionNameType.__args__),
    "house_type_name": list(HouseTypeNameType.__args__),
    "commercial_type_name": list(CommercialTypeNameType.__args__),
    "land_type_name": list(LandTypeNameType.__args__),
    "property_status_name": list(PropertyStatusNameType.__args__),
}


class PropertyFeatures(BaseModel):
    """
    Mô hình dữ liệu cho các đặc điểm bất động sản.
    Tất cả các trường đều là tùy chọn để cho phép thu thập dữ liệu từng bước.
    """

    # Địa điểm
    area_name: Optional[AreaNameType] = Field(None, description="Tên quận/huyện (ví dụ: Quận 1, Thành phố Thủ Đức)")
    longitude: Optional[float] = Field(None, description="Kinh độ")
    latitude: Optional[float] = Field(None, description="Vĩ độ")
    is_main_street: Optional[bool] = Field(None, description="Là nhà mặt phố/mặt tiền hay không")

    # Loại bất động sản
    category_name: Optional[CategoryNameType] = Field(None, description="Danh mục (Căn hộ/Chung cư, Nhà ở, Đất, Văn phòng)")
    house_type_name: Optional[HouseTypeNameType] = Field(None, description="Loại nhà (Nhà biệt thự, Nhà phố liền kề, Nhà ngõ hẻm, Nhà mặt phố mặt tiền)")
    apartment_type_name: Optional[ApartmentTypeNameType] = Field(None, description="Loại chung cư (Chung cư, Penthouse, Duplex, Officetel)")
    commercial_type_name: Optional[CommercialTypeNameType] = Field(None, description="Loại văn phòng/thương mại (Văn phòng, Shophouse, Mặt bằng kinh doanh)")
    land_type_name: Optional[LandTypeNameType] = Field(None, description="Loại đất (Đất thổ cư, Đất nền dự án, Đất nông nghiệp)")

    # Kích thước và Diện tích (m2)
    size: Optional[float] = Field(None, description="Diện tích đất/sử dụng (m2)")
    living_size: Optional[float] = Field(None, description="Diện tích sử dụng thực tế (m2)")
    width: Optional[float] = Field(None, description="Chiều ngang (m)")
    length: Optional[float] = Field(None, description="Chiều dài (m)")

    # Cấu trúc
    floors: Optional[int] = Field(None, description="Tổng số tầng")
    floornumber: Optional[int] = Field(None, description="Tầng số bao nhiêu (nếu là chung cư)")
    rooms_count: Optional[RoomsCountType] = Field(None, description="Số phòng ngủ (1-10, nhiều hơn 10)")
    toilets_count: Optional[ToiletsCountType] = Field(None, description="Số phòng vệ sinh (1-6, Nhiều hơn 6)")

    # Đặc điểm khác
    direction_name: Optional[DirectionNameType] = Field(None, description="Hướng nhà (Đông, Tây, Nam, Bắc, Đông Nam, Đông Bắc, Tây Nam, Tây Bắc)")
    balconydirection_name: Optional[BalconyDirectionNameType] = Field(None, description="Hướng ban công (Đông, Tây, Nam, Bắc, Đông Nam, Đông Bắc, Tây Nam, Tây Bắc)")
    furnishing_sell_status: Optional[FurnishingSellStatusType] = Field(None, description="Tình trạng nội thất cho BÁN (Nội thất đầy đủ, Nội thất cao cấp, Hoàn thiện cơ bản, Bàn giao thô)")
    furnishing_rent_status: Optional[FurnishingRentStatusType] = Field(None, description="Tình trạng nội thất cho THUÊ (Nội thất đầy đủ, Nội thất cao cấp, Hoàn thiện cơ bản, Bàn giao thô)")
    property_legal_document_status: Optional[PropertyLegalDocumentStatusType] = Field(None, description="Tình trạng pháp lý (Sổ hồng riêng, Đã có sổ, Đang chờ sổ, Hợp đồng mua bán)")

    # Rent-specific fields
    is_good_room: Optional[int] = Field(None, description="Đánh giá của nền tảng phòng tốt hay không (0 là không tốt, 1 là tốt) - CHỈ CHO THUÊ")
    deposit: Optional[float] = Field(None, description="Tiền cọc (VNĐ) - CHỈ CHO THUÊ")
    property_status_name: Optional[PropertyStatusNameType] = Field(None, description="Tình trạng bàn giao dự án (Đã bàn giao, Chưa bàn giao)")

    # Giá thực tế (từ tin đăng hoặc người dùng cung cấp)
    actual_price: Optional[float] = Field(None, description="Giá thực tế của bất động sản (VNĐ) - từ tin đăng hoặc người dùng cung cấp")

    class Config:
        json_schema_extra = {
            "example": {
                "area_name": "Quận 1",
                "category_name": "Nhà ở",
                "size": 100.5,
                "rooms_count": "3",
                "is_main_street": True
            }
        }
