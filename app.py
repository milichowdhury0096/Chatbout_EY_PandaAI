import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain_groq.chat_models import ChatGroq
import os
from pandasai import SmartDataframe

# Groq LLM Configuration
def load_groq_llm():
    return ChatGroq(model_name="llama3-groq-70b-8192-tool-use-preview", api_key=os.environ['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'])

# Main App Logic
def main():
    st.title("PandasAI with Groq and Streamlit")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        llm = load_groq_llm()
        df = SmartDataframe(data, config={'llm': llm})

        # Chat Interactions
        query = st.text_input("Enter your query about the data:")
        if query:
            response = df.chat(query)
            st.write(response)

if __name__ == "__main__":
    main()
