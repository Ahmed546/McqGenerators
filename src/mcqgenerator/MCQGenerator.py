import os
import json
import pandas as pd
from dotenv import load_dotenv

from src.mcqgenerator.logger import logging

# Import necessary libraries
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback


# Load environment variables
load_dotenv()

key = os.getenv("OPENAI_API_KEY")
print(key)
# Initialize OpenAI LLM
llm=ChatOpenAI(openai_api_key=key,model="gpt-3.5-turbo",temperature=0.7)

with open("D:\AI-Projects\MCQGenAI\McqGenerators\Response.json","r") as f:
    RESPONSE_JSON = json.load(f) 

# Prompt template
TEMPLATE = """
ext:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{RESPONSE_JSON}
"""

# Prompt
quiz_generation_prompt = PromptTemplate(
    input_variables= ["text","number","subject","tone","RESPONSE_JSON"],
    template=TEMPLATE
)

# quiz chain
quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt,output_key="quiz",verbose=True)


# review template
TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

# review prompt
quiz_evaluation_prompt = PromptTemplate(
    input_variables= ["subject","quiz"],
    template=TEMPLATE2
)

# review chain
review_chain =LLMChain(llm=llm, prompt=quiz_evaluation_prompt,output_key="review",verbose=True)


# combining the both chani sequences using sequential chain

generate_evaluation_chain = SequentialChain(chains=[quiz_chain,review_chain],
input_variables =["text","number","subject","tone","RESPONSE_JSON"],output_variables=["quiz","review"],verbose=True)
