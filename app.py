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

    # Display unknown fields
    if st.session_state.unknown_fields:
        st.header("Thông tin không rõ")
        st.write(", ".join(st.session_state.unknown_fields))

    if st.button("Làm mới cuộc trò chuyện"):
        st.session_state.messages = []
        st.session_state.features = PropertyFeatures()
        st.session_state.unknown_fields = []
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
            
            # Display AI response
            last_message = st.session_state.messages[-1]
            if last_message.type == "ai":
                with st.chat_message("assistant"):
                    st.write(extract_message_text(last_message))
            
            # Rerun to update sidebar
            st.rerun()
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
