import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

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

    # URL for the CSV file stored in your GitHub repository
    csv_url = "https://raw.githubusercontent.com/milichowdhury/Chatbout_EY_PandaAI/refs/heads/main/data/ai4i2020.csv"

    # Load the CSV data into a DataFrame
    data = pd.read_csv(csv_url)

    # Streamlit app layout
    st.title("AI4I 2020 Data Analysis with ChatGroq")

    # Define the prompt to guide the LLM's behavior
    instructions = """
    - Don't return data that is not in the table Machinelogs if asked to fetch data.
    - Machine failure 1 is machine failure and 0 means it is okay.
    - TWF is tool wear failure.
    - HDF is heat dissipation failure.
    - PWF is power failure.
    - OSF is overstrain failure.
    - RNF is random failures.
    - Only provide charts and graphs when necessary.
    - Provide data in a table format when necessary.
    """

    # Add a text input for chat-based interaction
    user_query = st.text_input("Ask a question related to the data:")

    # Process the user's query when they submit
    if st.button("Submit Query"):
        if user_query:
            # Send the query along with instructions to guide the model
            code_to_run = f"{user_query}. {instructions}"
            try:
                result = llm.chat(data, code_to_run)  # Assuming `llm.chat` is the method to run the query
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please enter a query to ask.")
