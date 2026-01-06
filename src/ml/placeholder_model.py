from src.models import PropertyFeatures
import random

class PricePredictor:
    """
    Mô hình dự đoán giá bất động sản (Placeholder).
    Sau này sẽ được thay thế bằng model Scikit-Learn hoặc XGBoost thực tế.
    """
    
    def __init__(self):
        pass

    def predict(self, features: PropertyFeatures) -> float:
        """
        Dự đoán giá dựa trên các đặc điểm.
        Hiện tại trả về một giá trị ngẫu nhiên hợp lý dựa trên diện tích.
        """
        base_price_per_m2 = 50_000_000  # 50 triệu/m2
        
        size = features.size if features.size else 0
        if size == 0:
            if features.living_size:
                size = features.living_size
            elif features.width and features.length:
                size = features.width * features.length
            else:
                # Giá trị default nếu không có size
                size = 50 
        
        # Điều chỉnh giá dựa trên vị trí (giả lập)
        location_factor = 1.0
        if features.area_name:
            if "Quận 1" in features.area_name:
                location_factor = 3.0
            elif "Thủ Đức" in features.area_name:
                location_factor = 1.2
            elif "Quận 7" in features.area_name:
                location_factor = 1.5
        
        estimated_price = size * base_price_per_m2 * location_factor
        
        # Thêm chút ngẫu nhiên để không bị cứng nhắc
        variation = random.uniform(0.9, 1.1)
        
        return estimated_price * variation
