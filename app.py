import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain_groq.chat_models import ChatGroq
import os
from pandasai import SmartDataframe

# Groq LLM Configuration
def load_groq_llm():
    return ChatGroq(model_name="llama3-groq-70b-8192-tool-use-preview", api_key=st.secrets['GROQ_API_KEY'])

# Main App Logic
def main():
    st.title("PandasAI with Groq and Streamlit")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview")
        st.dataframe(data.head())
        llm = load_groq_llm()
        df = SmartDataframe(data, config={'llm': llm})

        # Chat Interactions
        query = st.text_input("Enter your query about the data:")
        if query:
            response = df.chat(query)
                
                # Handling different types of responses
                if isinstance(response, pd.DataFrame):
                    st.write("Response in table format:")
                    st.dataframe(response)
                elif isinstance(response, str):
                    st.write("Response text:")
                    st.write(response)
                elif isinstance(response, plt.Figure):
                    st.write("Response is a plot:")
                    st.pyplot(response)
                else:
                    st.write("Unhandled response type.")
                    st.write(response)
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
