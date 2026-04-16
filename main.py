import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🌐 RAG Based Chatbot")

# Sidebar URLs
url1 = st.sidebar.text_input("URL-1")
url2 = st.sidebar.text_input("URL-2")
url3 = st.sidebar.text_input("URL-3")

placeholder = st.empty()

# Process URLs
if st.sidebar.button("Process URLs"):
    urls = [u for u in [url1, url2, url3] if u]

    if not urls:
        placeholder.text("Please enter at least one URL")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
query = st.chat_input("Ask something...")

if query:
    try:
        answer, sources = generate_answer(query)

        st.session_state.messages.append(("user", query))
        st.session_state.messages.append(
            ("bot", answer + "\n\nSources:\n" + sources)
        )

    except RuntimeError:
        placeholder.text("⚠️ Please click 'Process URLs' first")

# Display chat
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)