from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter


import os
import requests
import openai
import requests
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
from dotenv import load_dotenv

from tiktoken import Tokenizer, TokenizerV2
from tiktoken.models import Model
import textwrap

tokenizer = Tokenizer(Model())



import streamlit as st

load_dotenv()



# format your ouptut as a python list of strings, where each string is a link. e.g. ['https://link1', https://link2', 'https://link3, ...]

# Create a list of links that are relevant to the job postings
clean_links_template = """
SYSTEM:
Clean up these links so that the only ones remaining are ones that actually relate to job postings:

LINKS:
{links}

OUTPUT:"""


summariser_template = """
SYSTEM:
In this data from a job posting, please summarise the company name, job title, job description, location, the company website link, the time of the posting, and the url link of the job listing itself
Format your ouptut as a SINGLE python dictionary.


Example:
{{
"company_name": "Google", 
"job_title": "Sustainability Manager",
"job_description": "this is a job description", 
"location": "London", 
"website_link": "https://google.com",
"time_posted": "2 days ago"
"job_post_link": "https://www.seek.com/job/69",
}}

DATA:
{data}

OUTPUT:"""






clean_links_prompt = PromptTemplate(
    input_variables=["links"], template=clean_links_template,
)
summariser_prompt = PromptTemplate(
    input_variables=["data"], template=summariser_template
)



# Create a new instance and chain of the OpenAI chat model 
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=300,
)

links_llm_chain = LLMChain(prompt=clean_links_prompt, llm=llm, verbose=True)
summariser_llm_chain= LLMChain(prompt=summariser_prompt, llm=llm, verbose=True)




    
def get_links(query):
    # url = "https://www.linkedin.com/jobs/search/?currentJobId=3650756783&keywords=sustainability&refresh=true"
    response = requests.get(query)
    soup = BeautifulSoup(response.text, 'html.parser')
    # print(soup.prettify())
    job_postings_links = soup.find_all('a')
    
    # loop through the links and for each link check if it has the characters /jobs in it. if yes, then add it to the list of links
    relevant_links = []
    for link in job_postings_links:
        if link.get('href') is not None:
            if 'linkedin' in query:
                if '/jobs' in link.get('href'):
                    relevant_links.append(link.get('href'))
            elif 'seek' in query:
                if '/job/' in link.get('href'):
                    relevant_links.append('https://www.seek.com.au' + link.get('href'))
                
                
    # Remove dupliacte links
    relevant_links_refined = list(dict.fromkeys(relevant_links))
    # print(relevant_links_refined)
    num_results = len(relevant_links_refined)
    st.text(f'Number of job postings found {num_results}')
    for i in relevant_links_refined:
        print(i)
    
    return relevant_links_refined


       
               
def extract_content(links):
    counter = 0
    for link in links:
        loader = WebBaseLoader(link)
        docs = loader.load()
        answer = summarise(docs)
        df = pd.DataFrame([answer])
        if counter == 0:
            my_table = st.dataframe(df)
            df.to_csv('seek_job_postings.csv', index=False)  # Create a new CSV file for the first data frame.
        else:
            my_table.add_rows(df)
            df.to_csv('seek_job_postings.csv', mode='a', header=False, index=False)  # Append to the CSV file for subsequent data frames.

        counter+=1
        
       
        
def summarise(docs):
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=2500, chunk_overlap=100, length_function=len)
    # text = text_splitter.split_documents(docs)    
    chunks = split_into_chunks(docs)

    summary_string = ''
    for chunk in text:

        with get_openai_callback() as cb:
            answer = summariser_llm_chain.predict(data=chunk)
            summary_string = summary_string + '\n\n' + answer
        print(f"            Total number of tokens used: {cb.total_tokens}")
        
    answer = summariser_llm_chain.predict(data=summary_string)
    print("ANSWER")
    # print(answer)
    
    try:
        json_answer = json.loads(answer)
    except json.JSONDecodeError:
        # Attempt to fix invalid escape sequences and decode again
        fixed_answer = answer.encode('utf-8').decode('unicode_escape')
        try:
            json_answer = json.loads(fixed_answer)
        except json.JSONDecodeError:
            print("Could not decode JSON, even after fixing escape sequences.")
            return None

    
    print("JSON ANSWER")
    print(json_answer)
    return json_answer


def count_tokens(text):
    return len(list(tokenizer.tokenize(text)))

def split_into_chunks(text, max_tokens=2500):
    words = text.split(' ')
    chunks = []
    current_chunk = ''
    
    for word in words:
        if count_tokens(current_chunk + ' ' + word) <= max_tokens:
            current_chunk += ' ' + word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks



def main():
    
    st.set_page_config(page_title="Autonomous researcher - Job postings", page_icon="🥳")
    st.header("Web Scraper 🕸️")

    query = st.text_input("""Enter LinkedIn URL:
https://www.linkedin.com/jobs/search/?currentJobId=3650756783&keywords=sustainability&refresh=true

or Seek URL:
https://www.seek.com.au/sustainability-jobs""")

    if query:
        links = get_links(query)
        extract_content(links)
        
        
    


if __name__ == '__main__':
    main()