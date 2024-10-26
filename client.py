import requests
import streamlit as st

# API URL for your backend
API_URL = "http://localhost:8000/ask"

# Function to send question to the API
def ask_question(session_id: str, question: str):
    response = requests.post(API_URL, json={"session_id": session_id, "question": question})
    if response.status_code == 200:
        return response.json().get("answer")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

# Function to handle question submission
def handle_question_submission():
    question = st.session_state.question_input
    session_id = "sess_001"
    
    if question.lower() == "exit":
        st.write("Exiting the chatbot. Thank you!")
        st.stop()
    elif question.strip():
        # Append the user's question to the chat history
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Retrieve the answer from the backend API
        answer = ask_question(session_id, question)
        
        # If there's a valid response, append the answer to the chat history
        if answer:
            st.session_state.chat_history.append({"role": "bot", "content": answer})

def main():
    st.set_page_config(page_title="AI Q&A Chatbot", page_icon="📑", layout="centered")

    # Title and description
    st.markdown("<h1 style='text-align: center; color: #e4572e;'>AI-Powered Q&A Chatbot ✨</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Ask anything, and let the AI assist you!</p>", unsafe_allow_html=True)

    st.markdown("</br>", unsafe_allow_html=True)
    # Initialize chat history if it's not in the session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input field
    st.text_input(
        "Type your question:",
        placeholder="Ask me anything...",
        key="question_input",
        on_change=handle_question_submission
    )

    # Display the conversation 
    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        for message_data in st.session_state.chat_history:
            if message_data["role"] == "user":
                st.markdown(
                    f"<div class='user-message'>{message_data['content']}</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='bot-message'>{message_data['content']}</div>", 
                    unsafe_allow_html=True
                )

    # Custom CSS for chat UI
    st.markdown("""
        <style>
        .user-message {
            background-color: #ff6f59;
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .bot-message {
            background-color: #f2f3f7;
            color: black;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 70%;
            float: left;
            clear: both;
        }
        .stTextInput > div > input {
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
