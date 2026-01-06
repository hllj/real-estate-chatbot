import requests
from bs4 import BeautifulSoup
from src.models import PropertyFeatures

def fetch_property_details(url: str) -> PropertyFeatures:
    """
    Trích xuất thông tin bất động sản từ URL.
    Hỗ trợ các trang web bất động sản phổ biến ở Việt Nam (Basic Implementation).
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # TODO: Implement parsing logic for specific domains (batdongsan.com.vn, chotot.com, etc.)
        # Hiện tại trả về object rỗng để test flow
        
        # Mocking extraction for demo purposes if parsing fails or not implemented
        features = PropertyFeatures()
        
        # Example: Try to find title to guess category
        title = soup.title.string if soup.title else ""
        if "chung cư" in title.lower():
            features.category_name = "Căn hộ/Chung cư"
        elif "nhà" in title.lower():
            features.category_name = "Nhà ở"
            
        return features
        
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return PropertyFeatures()
