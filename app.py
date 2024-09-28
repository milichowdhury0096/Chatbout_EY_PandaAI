import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from pandasai import PandasAI
from pandasai.llm.chatgroq import ChatGroq

# Load environment variables from the .env file
load_dotenv()

# Get the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is loaded correctly
if not groq_api_key:
    st.error("GROQ_API_KEY not set. Please check your .env file.")
else:
    # Initialize the ChatGroq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview")
    # create PandasAI object, passing the LLM
    pandas_ai = PandasAI(llm)

    # URL for the CSV file stored in your GitHub repository
    csv_url = "https://raw.githubusercontent.com/milichowdhury/Chatbout_EY_PandaAI/refs/heads/main/data/ai4i2020.csv"

    # Load the CSV data into a DataFrame
    data = pd.read_csv(csv_url)
    prompt = st.text_area("Enter your prompt:")
    df = SmartDataframe(data, config={'llm': llm})

    # Streamlit app layout
    st.title(" Data Analysis with ChatGroq")


   
    # Generate output
    if st.button("Generate"):
        if prompt:
            # call pandas_ai.run(), passing dataframe and prompt
            with st.spinner("Generating response..."):
                st.write(pandas_ai.run(df, prompt))
        else:
            st.warning("Please enter a prompt.")

