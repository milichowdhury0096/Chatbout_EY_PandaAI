import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set your API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-groq-70b-8192-tool-use-preview")

# URL for the CSV file
csv_url = "https://raw.githubusercontent.com/milichowdhury/Chatbout_EY_PandaAI/refs/heads/main/data/ai4i2020.csv"

# Load the CSV data into a DataFrame
data = pd.read_csv(csv_url)
# Ensure SmartDataframe is initialized properly
df = SmartDataframe(data, config={'llm': llm})

# Streamlit app layout
st.title("AI4I 2020 Data Analysis with ChatGroq")

# Add a text input for chat-based interaction
user_query = st.text_input("Ask a question related to the data:")

# Process the user's query when they submit
if st.button("Submit Query"):
    if user_query:
        # Send the query along with instructions to guide the model
        result = df.chat(f"{user_query}. {instructions}")
        st.write(result)
    else:
        st.write("Please enter a query to ask.")
