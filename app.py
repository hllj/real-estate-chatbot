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
if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

# Display Sidebar for Debug/Info
with st.sidebar:
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
            st.metric("Giá dự đoán", f"{price:,.0f} VNĐ")

            # Show confidence interval
            confidence = prediction.get("confidence_interval_90")
            if confidence and len(confidence) == 2:
                lower, upper = confidence
                st.caption(f"Khoảng tin cậy 90%: {lower:,.0f} - {upper:,.0f} VNĐ")

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

    # Display unknown fields
    if st.session_state.unknown_fields:
        st.header("Thông tin không rõ")
        st.write(", ".join(st.session_state.unknown_fields))

    if st.button("Làm mới cuộc trò chuyện"):
        st.session_state.messages = []
        st.session_state.features = PropertyFeatures()
        st.session_state.unknown_fields = []
        if "prediction_result" in st.session_state:
            st.session_state.prediction_result = None
        st.rerun()

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
            
            # Display AI response
            last_message = st.session_state.messages[-1]
            if last_message.type == "ai":
                with st.chat_message("assistant"):
                    st.write(extract_message_text(last_message))
            
            # Rerun to update sidebar
            st.rerun()
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
