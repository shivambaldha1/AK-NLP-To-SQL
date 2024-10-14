import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import sqlite3
from langchain_community.utilities.sql_database import SQLDatabase
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import os

# OpenAI API Key
key = "sk-C2nXGLsgbZf98BNEau0X6YkjaOVLmnAlZ5Z1FalPh8T3BlbkFJpgZawjZgPzU_cEeoXWbTmnNzMPdkn_2vS0vxY5zvEA"


def database_from_sqlitefile(sql_file):
    """Read SQL file from local path, populate in-memory database, and create engine."""
    with open(sql_file, "r") as file:
        sql_script = file.read()

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

# Function to generate answer from SQL query
def generate_answer(db, table_info, question):
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                    api_key=key,
                    temperature=0)

    generate_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)

    answer_prompt = PromptTemplate.from_template(
    """
        **System:** You are an expert in MySQL. Based on the input question, generate a syntactically correct MySQL query. Follow any specific instructions provided.

        **Table Information:**  
        {table_info}

        **Input Question:**  
        {question}

        **Generated SQL Query:**  
        {query}

        **SQL Result:**  
        {result}

        **Final Answer:**  
    """
    )

    rephrase_answer = answer_prompt | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(query=generate_query).assign(
            result=itemgetter("query") | execute_query
        )
        | rephrase_answer
    )

    response = chain.invoke({
        "table_info": table_info,
        "question": question})
    
    return response

# Streamlit app
st.title("Chat With Your Data")
st.write("This chatbot allows you to ask questions about your SQL database.")

# Initialize database
engine = database_from_sqlitefile(sql_file="./SaaSSales.sql")
db = SQLDatabase(engine)
table_info = db.table_info

# Display database schema in the sidebar
with st.sidebar:
    st.header("Database Schema")
    st.code(table_info)

# Input question from user
if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.chat_input("Enter your question about the database:")

if prompt:
    try:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and execute SQL query
        response = generate_answer(db, table_info, prompt)

        # Display assistant response in chat message container
        with st.spinner("Generating response..."):
            with st.chat_message("assistant"):
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")