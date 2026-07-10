import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("MARYAMSHAHEEN11/mental-health-chatbot")
    model = AutoModelForCausalLM.from_pretrained("MARYAMSHAHEEN11/mental-health-chatbot")
    return tokenizer, model

tokenizer, model = load_model()

def chat_with_bot(user_input):
    prompt = f"User: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Bot:" in response:
        response = response.split("Bot:")[-1].strip()
    return response

st.set_page_config(page_title="Mental Health Support Chatbot", page_icon="💙")
st.title("💙 Mental Health Support Chatbot")
st.write("A supportive space to talk about stress, anxiety, and emotional wellness. Not a substitute for professional help.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Share what's on your mind...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chat_with_bot(user_input)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})