import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import create_graph
from src.models import PropertyFeatures
import pandas as pd


def extract_message_text(message) -> str:
    """Extract text content from AIMessage, handling both string and list formats."""
    content = message.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of content blocks (from tool-enabled LLM)
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)

# Page Config
st.set_page_config(page_title="Dự Đoán Giá Bất Động Sản", page_icon="🏠")

st.title("🏠 Trợ Lý Bất Động Sản AI")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "features" not in st.session_state:
    st.session_state.features = PropertyFeatures()
if "unknown_fields" not in st.session_state:
    st.session_state.unknown_fields = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "price_comparison" not in st.session_state:
    st.session_state.price_comparison = None
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()
if "mode" not in st.session_state:
    st.session_state.mode = "Sell"  # Default mode

# Mode Selection - Only show if conversation hasn't started
if len(st.session_state.messages) == 0:
    st.markdown("### Chọn chế độ dự đoán giá:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏷️ BÁN", use_container_width=True, type="primary" if st.session_state.mode == "Sell" else "secondary"):
            st.session_state.mode = "Sell"
            st.rerun()
    with col2:
        if st.button("🔑 CHO THUÊ", use_container_width=True, type="primary" if st.session_state.mode == "Rent" else "secondary"):
            st.session_state.mode = "Rent"
            st.rerun()

    # Display current mode
    mode_display = "**BÁN** (Dự đoán giá bán)" if st.session_state.mode == "Sell" else "**CHO THUÊ** (Dự đoán giá thuê/tháng)"
    st.info(f"Chế độ hiện tại: {mode_display}")
    st.markdown("---")

# Display Sidebar for Debug/Info
with st.sidebar:
    # New chat button at top
    if st.button("🔄 Làm mới cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.features = PropertyFeatures()
        st.session_state.unknown_fields = []
        st.session_state.mode = "Sell"  # Reset to default mode
        if "prediction_result" in st.session_state:
            st.session_state.prediction_result = None
        if "price_comparison" in st.session_state:
            st.session_state.price_comparison = None
        st.rerun()

    st.divider()

    # Show current mode at top of sidebar
    mode_label = "🏷️ BÁN" if st.session_state.mode == "Sell" else "🔑 CHO THUÊ"
    st.markdown(f"### Chế độ: {mode_label}")
    st.divider()

    st.header("Thông tin đã thu thập")
    features_dict = st.session_state.features.dict(exclude_none=True)
    if features_dict:
        st.json(features_dict)
    else:
        st.write("Chưa có thông tin.")

    # Display prediction result with SHAP explanation
    if "prediction_result" in st.session_state and st.session_state.prediction_result:
        prediction = st.session_state.prediction_result
        if prediction.get("predicted_price"):
            st.header("Kết quả dự đoán")
            price = prediction["predicted_price"]
            price_unit = "VNĐ" if st.session_state.mode == "Sell" else "VNĐ/tháng"
            st.metric("Giá dự đoán", f"{price:,.0f} {price_unit}")

            # Show confidence interval
            confidence = prediction.get("confidence_interval_90")
            if confidence and len(confidence) == 2:
                lower, upper = confidence
                st.caption(f"Khoảng tin cậy 90%: {lower:,.0f} - {upper:,.0f} {price_unit}")

            # Show SHAP explanation
            shap_explanation = prediction.get("shap_explanation")
            if shap_explanation and shap_explanation.get("success"):
                with st.expander("🔍 Phân tích giá chi tiết (SHAP)", expanded=False):
                    # Show base value
                    base_value = shap_explanation.get("base_value")
                    if base_value is not None:
                        base_price = 10 ** base_value
                        st.markdown(f"**Giá cơ sở (Base Value):** {base_price:,.0f} VNĐ")
                        st.caption(f"Log₁₀ base value: {base_value:.4f}")
                        st.divider()

                    # Get all features (use all_contributions for complete list)
                    all_features = shap_explanation.get("all_contributions", [])

                    if all_features:
                        # Separate positive and negative impacts (filter out near-zero values)
                        positive_impacts = [f for f in all_features if f.get("shap_value", 0) > 0.001]
                        negative_impacts = [f for f in all_features if f.get("shap_value", 0) < -0.001]

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**📈 Yếu tố làm TĂNG giá:**")
                            if positive_impacts:
                                for feat in positive_impacts:
                                    pct_impact = (10 ** feat["shap_value"] - 1) * 100
                                    vn_name = feat.get("feature_vn", feat["feature"])
                                    value = feat.get("feature_value")
                                    value_str = f" ({value})" if value is not None else ""
                                    st.markdown(f"- {vn_name}{value_str}: **+{pct_impact:.1f}%**")
                            else:
                                st.caption("Không có")

                        with col2:
                            st.markdown("**📉 Yếu tố làm GIẢM giá:**")
                            if negative_impacts:
                                for feat in negative_impacts:
                                    pct_impact = (1 - 10 ** feat["shap_value"]) * 100
                                    vn_name = feat.get("feature_vn", feat["feature"])
                                    value = feat.get("feature_value")
                                    value_str = f" ({value})" if value is not None else ""
                                    st.markdown(f"- {vn_name}{value_str}: **-{pct_impact:.1f}%**")
                            else:
                                st.caption("Không có")

                        # Show total features count
                        st.divider()
                        st.caption(f"Tổng số features: {len(all_features)} | Tăng giá: {len(positive_impacts)} | Giảm giá: {len(negative_impacts)}")

            # Indicate fallback model
            if prediction.get("is_fallback"):
                st.warning("⚠️ Sử dụng mô hình dự báo thay thế")

    # Display actual price if available
    actual_price = st.session_state.features.actual_price
    if actual_price:
        st.header("💰 Giá thực tế")
        from src.utils.price_comparison import format_price_vnd
        st.metric("Giá tin đăng", format_price_vnd(actual_price))

    # Display price comparison if available
    if st.session_state.price_comparison:
        comparison = st.session_state.price_comparison
        st.header("📊 So sánh giá")

        # Show accuracy level with color
        accuracy = comparison.get("accuracy_level", "")
        if accuracy == "Xuất sắc":
            st.success(f"🎯 Độ chính xác: {accuracy}")
        elif accuracy == "Tốt":
            st.success(f"✅ Độ chính xác: {accuracy}")
        elif accuracy == "Khá":
            st.info(f"📊 Độ chính xác: {accuracy}")
        else:
            st.warning(f"📈 Độ chính xác: {accuracy}")

        # Show difference
        diff_percent = comparison.get("difference_percent", 0)
        difference = comparison.get("difference", 0)
        st.metric(
            "Chênh lệch",
            f"{diff_percent:.1f}%",
            delta=f"{format_price_vnd(abs(difference))}",
            delta_color="normal" if difference >= 0 else "inverse"
        )

        # Show comparison text
        with st.expander("📝 Chi tiết so sánh", expanded=True):
            st.markdown(comparison.get("comparison_text_vn", ""))

    # Display unknown fields
    if st.session_state.unknown_fields:
        st.header("Thông tin không rõ")
        st.write(", ".join(st.session_state.unknown_fields))

# Display Chat History
for msg in st.session_state.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.write(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.write(extract_message_text(msg))

# Chat Input
if prompt := st.chat_input("Nhập thông tin bất động sản (VD: Nhà ở Quận 1, 50m2...)"):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Prepare state
    initial_state = {
        "messages": st.session_state.messages,
        "features": st.session_state.features,
        "mode": st.session_state.mode,
        "unknown_fields": st.session_state.unknown_fields
    }

    # Run graph
    with st.spinner("Đang xử lý..."):
        try:
            response = st.session_state.graph.invoke(initial_state)

            # Update state
            st.session_state.messages = response['messages']
            st.session_state.features = response.get('features', st.session_state.features)
            st.session_state.unknown_fields = response.get('unknown_fields', st.session_state.unknown_fields)
            st.session_state.prediction_result = response.get('prediction_result')
            st.session_state.price_comparison = response.get('price_comparison')
            
            # Display AI response
            last_message = st.session_state.messages[-1]
            if last_message.type == "ai":
                with st.chat_message("assistant"):
                    st.write(extract_message_text(last_message))
            
            # Rerun to update sidebar
            st.rerun()
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
