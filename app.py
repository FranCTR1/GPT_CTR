import streamlit as st
from query import generate_response

st.set_page_config(page_title="Consulta a CTR", page_icon="🔬")
st.title("🔎 CTR")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hazme una pregunta sobre materiales de laboratorio..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    response = generate_response(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
