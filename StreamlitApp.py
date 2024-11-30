import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluation_chain
from src.mcqgenerator.logger import logging


#loading json file
with open("Response.json","r") as f:
    RESPONSE_JSON = json.load(f)  


#creating title of the app
st.title="MCQ generation App with langchain and openAI"

with st.form("user input"):
    #file upload
    uploaded_file = st.file_uploader("upload a PDF or text file")

    #input fields
    mcq_content = st.number_input("No. of mcqs", min_value=3,max_value=5)

    #subject 
    subject = st.text_input("Insert Subject", max_chars=20)

    #Quiz tone 
    tone = st.selectbox("Quiz tone", ["easy", "medium", "hard"])

    # Add button
    button = st.form_submit_button("Create MCQs")


    # check if the button is clicked and all fields are filled

    if button and uploaded_file and mcq_content and subject and tone:
        with st.spinner("loading ......"):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    res = generate_evaluation_chain({
                        "text":text,
                        "number":mcq_content,
                        "subject":subject,
                        "tone":tone,
                        "RESPONSE_JSON":json.dumps(RESPONSE_JSON)
                        }
                    )

                #st.write(response)
                    
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("An error occurred while processing the file.")

            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(res, dict):
                    #extract the quiz data from the response
                    quiz= res.get("quiz",None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            #display the review in a text box as well
                            st.text_area(label="Review", value=res["review"])
                        else:
                            st.error("No MCQ data found in the response.")

                else:
                    st.write(res)
