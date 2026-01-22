import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import create_graph
from src.models import PropertyFeatures
from src.utils.model_downloader import download_models_if_missing, check_models_exist, MODELS
from src.utils.data_downloader import download_data_if_missing, check_data_files_exist, DATA_FILES
import pandas as pd


@st.cache_resource(show_spinner=False)
def ensure_models_downloaded():
    """
    Ensure ML models are downloaded. This runs once per session.
    Uses st.cache_resource to avoid re-downloading on every rerun.
    """
    model_status = check_models_exist()
    missing_models = [name for name, info in model_status.items() if not info["exists"]]

    if not missing_models:
        return True, "All models present"

    return False, missing_models


def download_missing_models(missing_models: list):
    """Download missing models with Streamlit progress UI."""
    st.info("🔄 Đang tải mô hình ML... (chỉ chạy một lần)")

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_models = len(missing_models)

    for idx, model_name in enumerate(missing_models):
        config = MODELS[model_name]
        status_text.text(f"Đang tải: {config['description']} ({model_name})...")

        def update_progress(name: str, downloaded: int, total: int):
            if total > 0:
                model_progress = downloaded / total
                overall_progress = (idx + model_progress) / total_models
                progress_bar.progress(overall_progress)
                size_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                status_text.text(f"Đang tải {name}: {size_mb:.1f}/{total_mb:.1f} MB")

        success, results = download_models_if_missing(progress_callback=update_progress)

    progress_bar.progress(1.0)
    status_text.text("✅ Hoàn tất tải mô hình!")

    # Clear progress indicators after a moment
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    return success


@st.cache_resource(show_spinner=False)
def ensure_data_downloaded():
    """
    Ensure data files are downloaded. This runs once per session.
    Uses st.cache_resource to avoid re-downloading on every rerun.
    """
    data_status = check_data_files_exist()
    missing_data = [name for name, info in data_status.items() if not info["exists"]]

    if not missing_data:
        return True, "All data files present"

    return False, missing_data


def download_missing_data(missing_files: list):
    """Download missing data files with Streamlit progress UI."""
    st.info("🔄 Đang tải dữ liệu BĐS... (chỉ chạy một lần)")

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(missing_files)

    for idx, file_name in enumerate(missing_files):
        config = DATA_FILES[file_name]
        status_text.text(f"Đang tải: {config['description']} ({file_name})...")

        def update_progress(name: str, downloaded: int, total: int):
            if total > 0:
                file_progress = downloaded / total
                overall_progress = (idx + file_progress) / total_files
                progress_bar.progress(overall_progress)
                size_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                status_text.text(f"Đang tải {name}: {size_mb:.1f}/{total_mb:.1f} MB")

        success, results = download_data_if_missing(progress_callback=update_progress)

    progress_bar.progress(1.0)
    status_text.text("✅ Hoàn tất tải dữ liệu!")

    # Clear progress indicators after a moment
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    return success


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

# Check and download models if needed (runs once per session)
models_ready, missing = ensure_models_downloaded()
if not models_ready:
    download_missing_models(missing)
    # Clear cache to re-check after download
    ensure_models_downloaded.clear()
    st.rerun()

# Check and download data files if needed (runs once per session)
data_ready, missing_data = ensure_data_downloaded()
if not data_ready:
    download_missing_data(missing_data)
    # Clear cache to re-check after download
    ensure_data_downloaded.clear()
    st.rerun()

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
if "listing_recommendations" not in st.session_state:
    st.session_state.listing_recommendations = None
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
        if "listing_recommendations" in st.session_state:
            st.session_state.listing_recommendations = None
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

    # Display listing recommendations
    if st.session_state.listing_recommendations and st.session_state.listing_recommendations.get("success"):
        recommendations = st.session_state.listing_recommendations
        listings = recommendations.get("listings", [])

        if listings:
            st.header("🏘️ BĐS tương tự")

            # Show search criteria
            criteria = recommendations.get("search_criteria", {})
            if criteria:
                with st.expander("📋 Tiêu chí tìm kiếm", expanded=False):
                    for key, value in criteria.items():
                        if value:
                            label_map = {
                                "khu_vuc": "Khu vực",
                                "loai_bds": "Loại BĐS",
                                "gia_muc_tieu": "Giá mục tiêu",
                                "dien_tich": "Diện tích",
                                "so_phong_ngu": "Số phòng ngủ"
                            }
                            st.markdown(f"**{label_map.get(key, key)}:** {value}")

            # Show relaxation info if any
            relaxation = recommendations.get("relaxation_applied")
            if relaxation:
                st.caption(f"📝 Đã điều chỉnh: {', '.join(relaxation)}")

            # Display each listing
            for idx, listing in enumerate(listings, 1):
                # Use subject as title if available
                subject = listing.get('subject', 'N/A')
                if subject != 'N/A' and len(subject) > 40:
                    display_subject = subject[:40] + "..."
                else:
                    display_subject = subject

                with st.expander(
                    f"**{idx}. {display_subject}** | {listing.get('do_tuong_dong', 'N/A')}",
                    expanded=idx == 1  # Expand first listing by default
                ):
                    # Show full subject/title
                    if subject != 'N/A':
                        st.markdown(f"**{subject}**")

                    st.markdown(f"📍 **{listing.get('loai_bds', 'BĐS')}** tại **{listing.get('khu_vuc', 'N/A')}**")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"💰 **Giá:** {listing.get('gia', 'N/A')}")
                        st.markdown(f"📐 **Diện tích:** {listing.get('dien_tich', 'N/A')}")
                        if listing.get('so_phong_ngu') != 'N/A':
                            st.markdown(f"🛏️ **Phòng ngủ:** {listing.get('so_phong_ngu')}")
                        if listing.get('so_toilet') != 'N/A':
                            st.markdown(f"🚿 **Toilet:** {listing.get('so_toilet')}")

                    with col2:
                        if listing.get('so_tang') != 'N/A':
                            st.markdown(f"🏢 **Số tầng:** {listing.get('so_tang')}")
                        if listing.get('huong') != 'N/A':
                            st.markdown(f"🧭 **Hướng:** {listing.get('huong')}")
                        if listing.get('loai_nha') != 'N/A':
                            st.markdown(f"🏠 **Loại:** {listing.get('loai_nha')}")
                        if listing.get('phap_ly') != 'N/A':
                            st.markdown(f"📋 **Pháp lý:** {listing.get('phap_ly')}")

                    # Similarity score indicator
                    score = listing.get('similarity_score', 0)
                    if score >= 70:
                        st.success(f"⭐ Độ tương đồng: **{listing.get('do_tuong_dong', 'N/A')}**")
                    elif score >= 50:
                        st.info(f"⭐ Độ tương đồng: **{listing.get('do_tuong_dong', 'N/A')}**")
                    else:
                        st.warning(f"⭐ Độ tương đồng: **{listing.get('do_tuong_dong', 'N/A')}**")

                    # Show URL link if available
                    if listing.get('url'):
                        st.markdown(f"🔗 [Xem chi tiết trên Nhà Tốt]({listing.get('url')})")

            st.caption(f"Tìm thấy {len(listings)} BĐS tương tự (khoảng giá ±{recommendations.get('final_price_range_pct', 0)}%)")

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
            # Update listing recommendations if present
            if response.get('listing_recommendations'):
                st.session_state.listing_recommendations = response.get('listing_recommendations')
            
            # Display AI response
            last_message = st.session_state.messages[-1]
            if last_message.type == "ai":
                with st.chat_message("assistant"):
                    st.write(extract_message_text(last_message))
            
            # Rerun to update sidebar
            st.rerun()
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
