# User Collected Features

This document outlines all feature data points collected from users for property price prediction, based on the codebase definitions.

## Feature Definitions

| Feature Name | Type | Description |
|:---|:---|:---|
| `area_name` | String (Category) | Tên quận/huyện (ví dụ: Quận 1, Tp Thủ Đức) |
| `longitude` | Float | Kinh độ |
| `latitude` | Float | Vĩ độ |
| `is_main_street` | Boolean | Là nhà mặt phố/mặt tiền hay không |
| `category_name` | String (Category) | Danh mục (ví dụ: Căn hộ/Chung cư, Nhà ở, Đất) |
| `house_type_name` | String (Category) | Loại nhà (ví dụ: Nhà biệt thự, Nhà phố liền kề) |
| `apartment_type_name` | String (Category) | Loại chung cư (ví dụ: Chung cư, Penthouse) |
| `commercial_type_name` | String (Category) | Loại văn phòng/thương mại |
| `land_type_name` | String (Category) | Loại đất |
| `size` | Float | Diện tích đất/sử dụng (m2) |
| `living_size` | Float | Diện tích sử dụng thực tế (m2) |
| `width` | Float | Chiều ngang (m) |
| `length` | Float | Chiều dài (m) |
| `floors` | Integer | Tổng số tầng |
| `floornumber` | Integer | Tầng số bao nhiêu (nếu là chung cư) |
| `rooms_count` | Integer | Số phòng ngủ |
| `toilets_count` | Integer | Số phòng vệ sinh |
| `direction_name` | String (Category) | Hướng nhà (Đông, Tây, Nam, Bắc...) |
| `balconydirection_name` | String (Category) | Hướng ban công |
| `furnishing_sell_status` | String (Category) | Tình trạng nội thất (Nội thất đầy đủ, Thô...) |
| `property_legal_document_status` | String (Category) | Tình trạng pháp lý (Sổ hồng, Hợp đồng...) |
| `property_status_name` | String (Category) | Tình trạng bàn giao dự án |

## Valid Values for Categorical Features

```json
{
    "area_name": [
        "Thành phố Thủ Đức",
        "Quận 7",
        "Quận Tân Phú",
        "Quận 11",
        "Huyện Củ Chi",
        "Quận 8",
        "Quận Bình Tân",
        "Quận 12",
        "Quận 5",
        "Huyện Bình Chánh",
        "Huyện Hóc Môn",
        "Quận Gò Vấp",
        "Quận Bình Thạnh",
        "Huyện Cần Giờ",
        "Quận Tân Bình",
        "Quận 10",
        "Huyện Nhà Bè",
        "Quận 1",
        "Quận Phú Nhuận",
        "Quận 6",
        "Quận 4",
        "Quận 3"
    ],
    "category_name": [
        "Căn hộ/Chung cư",
        "Đất",
        "Văn phòng, Mặt bằng kinh doanh",
        "Nhà ở"
    ],
    "is_main_street": [
        false,
        true
    ],
    "apartment_type_name": [
        "Chung cư",
        "Duplex",
        "Không có thông tin",
        "Căn hộ dịch vụ, mini",
        "Officetel",
        "Tập thể, cư xá",
        "Penthouse"
    ],
    "property_legal_document_status": [
        "Hợp đồng mua bán",
        "Sổ hồng riêng",
        "Đã có sổ",
        "Đang chờ sổ",
        "Hợp đồng đặt cọc",
        "Không có thông tin",
        "Giấy tờ khác"
    ],
    "rooms_count": [
        "2",
        "3",
        "Không có thông tin",
        "1",
        "nhiều hơn 10",
        "4",
        "5",
        "8",
        "6",
        "7",
        "10",
        "9"
    ],
    "toilets_count": [
        "2",
        "Không có thông tin",
        "1",
        "Nhiều hơn 6",
        "3",
        "5",
        "4",
        "6"
    ],
    "furnishing_sell_status": [
        "Không có thông tin",
        "Nội thất đầy đủ",
        "Nội thất cao cấp",
        "Hoàn thiện cơ bản",
        "Bàn giao thô"
    ],
    "balconydirection_name": [
        "Tây Bắc",
        "Không có thông tin",
        "Đông",
        "Tây Nam",
        "Nam",
        "Tây",
        "Đông Nam",
        "Bắc",
        "Đông Bắc"
    ],
    "direction_name": [
        "Đông Nam",
        "Không có thông tin",
        "Tây",
        "Đông Bắc",
        "Nam",
        "Tây Bắc",
        "Đông",
        "Tây Nam",
        "Bắc"
    ],
    "house_type_name": [
        "Không có thông tin",
        "Nhà ngõ, hẻm",
        "Nhà mặt phố, mặt tiền",
        "Nhà phố liền kề",
        "Nhà biệt thự"
    ],
    "commercial_type_name": [
        "Không có thông tin",
        "Mặt bằng kinh doanh",
        "Shophouse",
        "Văn phòng",
        "Officetel"
    ],
    "land_type_name": [
        "Không có thông tin",
        "Đất thổ cư",
        "Đất nền dự án",
        "Đất nông nghiệp",
        "Đất công nghiệp"
    ],
    "property_status_name": [
        "Chưa bàn giao",
        "Đã bàn giao",
        "Không có thông tin"
    ]
}
```
