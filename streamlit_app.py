import streamlit as st
from faq_bot import FaqBot
import logging
import time

st.set_page_config(page_title="FAQ Bot", layout="wide")

@st.cache_resource
def load_bot():
    with st.spinner("Initializing Bot, please wait..."):
        try:
            bot = FaqBot(model_dir='model_assets')
            bot.load()
            return bot
        except Exception as e:
            logging.error(f"Failed to load bot: {e}")
            return None

st.title("🤖 Mental Help Bot")

if st.button("Clear Chat / Start New"):
    st.session_state.messages = []
    st.session_state.chat_ended = False
    st.rerun()

st.write("Ask a question about common mental health topics, and the bot will try to find a relevant answer from its knowledge base." \
" Please describe your complete question in a single text so that the bot can provide the best possible solution.")

bot = load_bot()

EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye", "stop"}

if not bot:
    st.error("The bot could not be loaded. Please ensure the model assets exist and the server is configured correctly.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_ended" not in st.session_state:
        st.session_state.chat_ended = False
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.chat_ended:
        st.success("Thanks for using the bot. Developed by Tanishq with love <3")
        st.info("Please refresh the page to start a new chat.")
    else:
        if prompt := st.chat_input("What is your question? (Type 'quit' to exit)"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            if prompt.lower().strip() in EXIT_COMMANDS:
                st.session_state.chat_ended = True
                st.rerun()
            else:
                with st.spinner("Thinking..."): # This is already in the user's new code
                    response = bot.get_answer(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# st.caption("Developed by Tanishq with love <3")
footer_style = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: #808080; /* A subtle, medium-grey color */
    font-size: 0.9em; /* Make the font slightly smaller */
    padding-bottom: 5px; /* Add a little space from the screen's bottom edge */
    
    /* --- Crucial properties for this design --- */
    z-index: 999; /* Ensures it's on top */
    pointer-events: none; /* Makes the footer non-interactive, so clicks go through */
}
</style>
"""

footer_html = """
<div class="footer">
    <p>Developed by Tanishq with love <3</p>
</div>
"""

st.markdown(footer_style, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)