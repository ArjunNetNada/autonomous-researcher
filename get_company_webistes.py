from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import os
import requests
import psycopg2
import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain.utilities import GoogleSerperAPIWrapper

from dotenv import load_dotenv
load_dotenv()




import traceback
import time




# This is the template for the website retrieval task
website_template = """
What are the home page website links for these companies. 
Note, these companies are predominantly Australian based and if you are unable to find a website link for a company, please reply with 'Unable to find website link'


FORMAT INSTRUCTIONS: 
Format your response with the name of the company and website link like so:
```
Pepsi: https://www.pepsico.com/
Optus: https://www.optus.com.au/
Bar Reggio: http://www.barreggio.com.au/
Minor Adjustment: Unable to find website link
```


COMPANIES:
{companies}


OUPTUT:"""


website_prompt = PromptTemplate(
    input_variables=["companies"], template=website_template
)


# Create a new instance and chain of the OpenAI chat model 
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=300,
)

website_llm_chain = LLMChain(prompt=website_prompt, llm=llm, verbose=True)






def process_dataframe(df, column_name):
    batch_counter = 1
    batch_size = 50

    # Create a new column for the websites
    df['website'] = ''

    company_names = df[column_name].tolist()

    for i in range(0, len(company_names), batch_size):
        print(f"Processing batch {batch_counter}: {i}-{i+batch_size}")
        
        batch_company_names = company_names[i: i + batch_size]

        batch_string = ""
        for company in batch_company_names:
            batch_string += f"{company}\n"

        answer = get_company_website(batch_string)
        company_dict = extract_websites(answer)

        for i in range(len(batch_company_names)):
            if batch_company_names[i] in company_dict:
                df.loc[df[column_name] == batch_company_names[i], 'website'] = company_dict[batch_company_names[i]]

        batch_counter += 1

    return df


# OpenAI API call to retrieve the website of the company
def get_company_website(formatted_string):
    print(f"            Getting websites...")

    with get_openai_callback() as cb:
        answer = website_llm_chain.predict(companies=formatted_string)
        print(f"            Total number of tokens used: {cb.total_tokens}")
        
    return answer
    
    
# Extract the website from the answer 
def extract_websites(text):
    print(f"            Extracting websites...")

    lines = text.strip().split('\n')
    company_dict = {}
    for line in lines:
        if line.strip() == '' or ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        company_dict[key] = value

    return company_dict
    


def main():
    df = pd.read_csv('seek_job_postings.csv')
    df = process_dataframe(df, 'company_name')
    df.to_csv('seek_job_postings - with websites.csv', index=False)

if __name__ == '__main__':
    main()