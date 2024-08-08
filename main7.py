import pandas as pd
import streamlit as st
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

def main():
    load_dotenv()

    if os.getenv('GROQ_API_KEY') is None or os.getenv('GROQ_API_KEY') == "":
        st.error("GROQ_API_KEY is not set")
        return

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    if "Boston.csv" is not None:
        agent = create_csv_agent(
            ChatGroq(temperature=0.4), "Boston.csv", verbose=True, allow_dangerous_code=True, max_iterations=50,model="llama-3.1-405b-reasoning"
        )

        user_question = st.chat_input("Ask a question about your CSV: ")

        if user_question:
            with st.spinner(text="In progress..."):
                try:
                    response = agent.invoke(user_question)
                    st.write(response)
                except ValueError as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

