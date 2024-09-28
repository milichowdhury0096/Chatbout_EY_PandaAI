import os
import pandas as pd
os.system('pip install -r https://raw.githubusercontent.com/milichowdhury/Chatbout_EY_PandaAI/refs/heads/main/requirements.txt')
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set your API key
groq_api_key = os.getenv("GROQ_API_KEY", "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")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview")

# CSV file URL
csv_url = "https://raw.githubusercontent.com/milichowdhury/Chatbout_EY_PandaAI/refs/heads/main/data/ai4i2020.csv"

# Load the data
data = pd.read_csv(csv_url)
df = SmartDataframe(data, config={'llm': llm})

# Streamlit App
st.title("AI4I 2020 Data Analysis with ChatGroq")
