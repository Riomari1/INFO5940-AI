import streamlit as st

st.set_page_config(page_title="Hello Karam", layout="centered")

st.title("👋 Hello!")

if "messages" not in st.session_state:
    st.sesion_state("messages") = [{"role": "assistant", "content": "Marhaba!"}]

for msg in st.session_state:
    st.chat_message(msg['role']).write(msg['content'])
    
if prompt := st.chat_input():
    client = OpenAI
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(model = "openai.gpt-4o",
                                                )
name = st.text_input("What is your name?")
if name:
    st.success(f"Nice to meet you, {name}!")

st.markdown("---")
st.caption("This is a test app for INFO 5940 Fall 2025.")


