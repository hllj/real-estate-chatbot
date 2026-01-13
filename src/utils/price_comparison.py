"""
Utility functions for comparing predicted prices vs actual prices.
"""
from typing import Optional
from src.graph.state import PriceComparison


def format_price_vnd(price: float) -> str:
    """
    Format price in Vietnamese style.

    Args:
        price: Price in VND

    Returns:
        Formatted string (e.g., "5.2 tỷ", "850 triệu")
    """
    if price >= 1_000_000_000:
        value = price / 1_000_000_000
        if value == int(value):
            return f"{int(value)} tỷ"
        return f"{value:.1f} tỷ"
    elif price >= 1_000_000:
        value = price / 1_000_000
        if value == int(value):
            return f"{int(value)} triệu"
        return f"{value:.1f} triệu"
    else:
        return f"{price:,.0f} VNĐ"


def calculate_price_comparison(
    predicted_price: float,
    actual_price: float
) -> PriceComparison:
    """
    Calculate comparison metrics between predicted and actual prices.

    Args:
        predicted_price: The model's predicted price in VND
        actual_price: The actual listing price in VND

    Returns:
        PriceComparison with all metrics and Vietnamese explanation
    """
    difference = actual_price - predicted_price
    difference_percent = abs(difference / actual_price) * 100

    # Determine accuracy level
    if difference_percent < 10:
        accuracy_level = "Xuất sắc"
        accuracy_emoji = "🎯"
    elif difference_percent < 20:
        accuracy_level = "Tốt"
        accuracy_emoji = "✅"
    elif difference_percent < 30:
        accuracy_level = "Khá"
        accuracy_emoji = "📊"
    else:
        accuracy_level = "Cần cải thiện"
        accuracy_emoji = "📈"

    # Generate Vietnamese explanation
    predicted_formatted = format_price_vnd(predicted_price)
    actual_formatted = format_price_vnd(actual_price)
    diff_formatted = format_price_vnd(abs(difference))

    if difference > 0:
        direction = "thấp hơn"
        insight = "Có thể đây là cơ hội đầu tư tốt nếu giá thị trường hợp lý, hoặc bất động sản có những yếu tố đặc biệt chưa được tính trong mô hình."
    elif difference < 0:
        direction = "cao hơn"
        insight = "Giá tin đăng có thể thấp hơn giá trị thực tế, hoặc có thể người bán đang cần bán gấp."
    else:
        direction = "bằng"
        insight = "Dự đoán hoàn toàn chính xác!"

    comparison_text = f"""
{accuracy_emoji} **Độ chính xác: {accuracy_level}** (chênh lệch {difference_percent:.1f}%)

📌 **Giá dự đoán:** {predicted_formatted}
📌 **Giá thực tế:** {actual_formatted}
📌 **Chênh lệch:** {diff_formatted} ({direction} giá thực tế)

💡 **Nhận xét:** {insight}
""".strip()

    return PriceComparison(
        predicted_price=predicted_price,
        actual_price=actual_price,
        difference=difference,
        difference_percent=difference_percent,
        accuracy_level=accuracy_level,
        comparison_text_vn=comparison_text
    )


def get_comparison_if_available(
    predicted_price: Optional[float],
    actual_price: Optional[float]
) -> Optional[PriceComparison]:
    """
    Get price comparison if both prices are available.

    Args:
        predicted_price: The model's predicted price (may be None)
        actual_price: The actual listing price (may be None)

    Returns:
        PriceComparison if both prices available, None otherwise
    """
    if predicted_price is not None and actual_price is not None:
        if predicted_price > 0 and actual_price > 0:
            return calculate_price_comparison(predicted_price, actual_price)
    return None
