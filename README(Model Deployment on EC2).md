ubuntu@ip-10-0-3-242:~$ cat > app.py << 'EOF'
import streamlit as st
import requests
import json

st.set_page_config(page_title="IoT Security Chatbot", page_icon="🛡️")
st.title("🛡️ IoT Cybersecurity Analyst")
st.markdown("Analyze IoT network traffic for potential cyber attacks")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter network traffic details to analyze..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "cybersecurity-model", "prompt": prompt, "stream": False},
                timeout=30
            )
            reply = response.json().get("response", "Sorry, I couldn't analyze that.")
            st.markdown(reply)
        except Exception as e:
            st.markdown(f"Error: {e}")
EOF st.session_state.messages.append({"role": "assistant", "content": reply})
ubuntu@ip-10-0-3-242:~$ /home/ubuntu/.local/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.0.3.242:8501
  External URL: http://3.92.74.249:8501
