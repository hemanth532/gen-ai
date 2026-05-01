import streamlit as st
from ollama import chat

st.set_page_config(page_title="Qwen Chat", layout="wide")
st.title("💬 Chat with Qwen")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from Qwen
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(
                model='qwen3.5:0.8b',
                messages=[{"role": msg["role"], "content": msg["content"]} 
                         for msg in st.session_state.messages],
            )
            assistant_message = response.message.content
            st.markdown(assistant_message)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
