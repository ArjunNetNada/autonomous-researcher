from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains.summarize import load_summarize_chain




import os
import requests
import openai
import requests
from bs4 import BeautifulSoup
import time
import json
import pandas as pd
from dotenv import load_dotenv
import http.client
import traceback
import tiktoken
import streamlit as st

load_dotenv()





# Define the prompt template for the initial summary
question_prompt_template = """In this data from a company's carbon neutral page, please summarise the steps the company is taking to reduce their emissions and be more carbon neutral.
Please include the company name, the steps they are taking, and the link or links to the carbon neutral page(s) if they exist, and the date of when the carbon report was published.

DATA:
{text}

OUTPUT:"""



# Define the prompt template for refining the summary
refine_template = """Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {existing_answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary."""


# Define the prompt template for formatting output to json
format_template = """
SYSTEM:
In this data from a company's carbon neutral page, please summarise the steps the company is taking to reduce their emissions and be more carbon neutral.
Please include the company name, the steps they are taking, and the link or links to the carbon neutral page(s) if they exist, and the date of when the carbon report was published.
Format your ouptut as a SINGLE python dictionary.

Example:
{{
"company_name": "Google", 
"steps_taken": "This is a step. This is another step. This is a third step", 
"carbon_neutral_page_link": "https://google.com/carbon-neutral",
"published_date": "23/09/2021",
}}


DATA:
{data}

OUTPUT:"""


question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text"],)
refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_template,)
format_prompt = PromptTemplate(input_variables=["data"], template=format_template)


# Create a new instance and chain of the OpenAI chat model 
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    request_timeout=300,
)


# Load the refine chain
summarise_chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    return_intermediate_steps=True,
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    verbose=True,
)


format_chain= LLMChain(prompt=format_prompt, llm=llm, verbose=True)






def get_carbon_page_link(query):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": query,
    "gl": "au"
    })
    headers = {
    'X-API-KEY': os.environ["SERPAPI_API_KEY"],
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    result = data.decode("utf-8")
    json_result = json.loads(result)
    # title = json_result['organic'][0]['title']
    link = json_result['organic'][0]['link']
    # st.write(f"Webpage title: '{title}'")
    # st.write(f"Webpage link: '{link}'")
    return link


def summarise_carbon_page(link):
    try:
        if link.lower().endswith(".pdf"):
            loader_pdf = PyPDFLoader(link)
            docs = loader_pdf.load()
            # print("PDF LOADER")
        else: 
            loader = WebBaseLoader(link)
            docs = loader.load()
            # print("WEBPAGE LOADER")
    
    except:
        docs = []
        
    answer = summarise(docs)
    
    return answer

    
    
    
def summarise(docs):
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=6000, chunk_overlap=100, length_function=len)
    text = text_splitter.split_documents(docs)    
    
    if len(text) == 0:
        json_answer = ''
        return json_answer
        
    # print("TEXT")
    # print(len(text))
    # print(text)
    # print("TEXT")

    # summary_string = ''
    # for chunk in text:
    #     # print(type(chunk))
    #     # print(chunk)
    #     if num_tokens_from_string(str(chunk), "cl100k_base") > 3000:
    #         chunk = ''

    #     with get_openai_callback() as cb:
    #         answer = format_chain.predict(data=chunk)
    #         summary_string = summary_string + '\n\n' + answer
    #     print(f"            Total number of tokens used: {cb.total_tokens}")

    # Retrieve the final summary
    output = summarise_chain({"input_documents": text}, return_only_outputs=True)
    intermediate_answer = output["output_text"]
    # Pass summary to formatter
    answer = format_chain.predict(data=intermediate_answer)





    # print("ANSWER")
    # print(answer)
    json_answer = ''
    try:
        json_answer = json.loads(answer)
    except json.JSONDecodeError:
        # Attempt to fix invalid escape sequences and decode again
        fixed_answer = answer.encode('utf-8').decode('unicode_escape')
        try:
            json_answer = json.loads(fixed_answer)
        except json.JSONDecodeError:
            print("Could not decode JSON, even after fixing escape sequences.")
            print(traceback.format_exc())
            json_answer = ''

            return json_answer

    
    print("JSON ANSWER")
    print(json_answer)
    return json_answer

    # return answer
    
def num_tokens_from_string(text, encoding_name):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def main():
    
    st.set_page_config(page_title="Autonomous researcher - Carbon Neutrality", page_icon="🌿")
    st.header("Carbon Neutral Scraper 🌿")
    # query = st.text_input("Enter the search query: ", "O'Brien Glass Industries Limited carbon neutral")
    contact_names = ['Accor','Adobe','Adavale Country','Allianz','AMEX','Andrew Hutton','Ampol','Ampol Retail','Andrew Hutton (Expenses)','Ariba','AS Colour','ATO','Astronaut','Avetta','Babylon Newcastle','Balance Collective','Bar Beach Bowling Club','Barton Flower Barn','Belinda Rowan','Belflora','Billy Goat Catering','Blackbutt Supermarket','Blue Door Kiosk','Body Beyond Limits','BP','BP Bennetts','BP Whitebridge','Bp Newcastle West','BWS','Burwood Inn','Cabcharge Australia','Card.Gift','Cardiff Nissan','CBIC','Charlestown square','Chilli cake Wolli Creek','Christopher Cooper','Coles express','Country Trucker Caps','Crowne Plaza Hunter Valley','Crystalbrook Kingsley','Cuppa Joe\'s','Darks Coffee Roasters','Dash of Quay Brisbane City','Death & Taxes','Designer Delights','Digital River','Dewdrop Creative','Eastlakes Security','EasyPark','Energy Australia','Esri Australia','Exemplar Global','Experienced Office Furniture','Express Bi-fold Doors','Finbiz','Foodworks Whitebridge','Frost Security Locksmiths','Fume Hood Cleaning Australia','Gateshead car wash','Gateshead Car & Dog Wash','Gmcabs Australia','GM Taxis','Groupmap','Harry Hartog','Hawthorne club','HSBC','Howard Smith Wharves','Icare','Iga','IGA Merewether','IEMA','Jewells Cleaning Services','Jessica Southgate','Je Fleur Flowers','John Trotter','Kaden Centre','Kathmandu','Lake Macquarie City Council','Living Moments Photography','Lively Group','Mary Ellen','Market Expresso Cafe','Merewether','Meriton Suites - Brisbane','Metro Charlestown','Metro Petroleum Bellbird','Millennium Convenience Store','Microsoft Project','Mitsuki','Morning Market','Multiple Sclerosis Society','Myer','Nathan Archer','Natasha Thompson','Nearmaps','Nerida Manley','Newcastle Airport','NRMA','Noah\'s Mid City Motor Inn Muswellbrook','Peter James Madden','Pegasus','Peter James Madden','Pizza Cutters','Pizza Hut','Planet Fitness','Planning Institute Australia','Poolwerx','Prince Newcastle','Rapid Antigen Australia','Roberts Lawyers','RMS EToll','Rose Pascoe','Sanokil','Selina Brisbane','Skildare','Sophie Nicholas','Spotto','St Andrews House','St George','St George Fee','Stella Hand Car Wash','Suchai','Super What Not','Swim Safe Swim School','Synthesia','Talulah Café','Target','Telstra','The Cheese Cake Shop Kotara','The executive connection','The Good Guys','The Grain Thai','The Grain Thai Restaurant','The Greyhounds The Gardens','The Little Garden','Think-Write','Tirtyl','Totally Workwear','Treasury Casino & Hotel','UberX','UBER','Ultra Boolaroo','Unity','Uprising Bakery','Woolworths Charlestown','World Bicycle Relief','Worrimi Framing','Yellowbox HR Services','Yello Mechanical Services','Yoo Moo Services','Yabadoo Kids Entertainment','Zenith Plant Hire','Zippy Cleaning & Maintenance Services','Zero Latency','Zip Australia']

    
    
    if st.button("Search"):
        counter = 0
        for name in contact_names:
            print(f"Company Name: {name}")
            # extract carbon neutrality information
            try:
                query = name + " - carbon neutral" 
                link = get_carbon_page_link(query)
                print(f"Link: {link}")
                answer = summarise_carbon_page(link)
                print(f"Answer: {answer}")
                if answer == '':
                    answer = {
                    "company_name": name, 
                    "steps_taken": "N/A",
                    "carbon_neutral_page_link": "N/A",
                    "published_date": "N/A",
                    "google_link": link,
                    }
                else:   
                    answer["company_name"] = name
                    answer["google_link"] = link
            except:
                   answer = {
                    "company_name": name, 
                    "steps_taken": "N/A",
                    "carbon_neutral_page_link": "N/A",
                    "published_date": "N/A",
                    "google_link": "N/A",
                    }
            
            # extract contact information

                
            df = pd.DataFrame([answer])
            if counter == 0:
                my_table = st.dataframe(df)
                df.to_csv('carbon_report.csv', index=False)  # Create a new CSV file for the first data frame.
            else:
                my_table.add_rows(df)
                df.to_csv('carbon_report.csv', mode='a', header=False, index=False)  # Append to the CSV file for subsequent data frames.
                
            counter+=1
            print("----------------------------------------")

        # st.write(answer)
        # print(answer)
        
        st.write("Done!")
        print("Done!")
        return
        
        

    # do a google search with {company name} + "carbon neutral" and get the first result
    # web doc loader that url 
    # llm to summarise the article
    
        
    


if __name__ == '__main__':
    main()
    
    
    # # TEST STUFF
    # query = "Adobe - carbon neutral"
    # link = get_carbon_page_link(query)
    # print(f"Link: {link}")  
    # answer = summarise_carbon_page(link)
    # print(f"Answer: {answer}")
