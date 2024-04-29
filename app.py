
import streamlit as st
from dotenv import load_dotenv
import os
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
#from langchain.llms import OpenAI
#from langchain.callbacks import get_openai_callback
#from langchain.llms import HuggingFaceHub
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb


# Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot''')
    add_vertical_space(5)
    st.write('Made by Saurabh')

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200
            )
        chunks = text_splitter.split_text(text)
        return chunks


def get_vector_store(text_chunks, pdf_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorStore = FAISS.from_texts(text_chunks, embedding=embeddings)
     
    store_name = pdf_name[:-4]
 
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vectorStore = pickle.load(f)
        st.write('Embeddings Loaded from the Disk')
    else:
        # embeddings = OpenAIEmbeddings()
        #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorStore = FAISS.from_texts(text_chunks, embedding=embeddings)

        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorStore, f)
        st.write('Embeddings Saved to the Disk')
    return vectorStore

def get_conversational_chain():
    prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
            if the answer is not in the provided context then just say, "answer is not available in the context", don't provide the wrong answer\n\n
            Context:\n {context}? \n
            Question:\n {question} \n

            Answer:
        """

    llm = ChatGoogleGenerativeAI(model = "gemini-pro", temperature=0.3)
        
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)
 
def main():
    
    load_dotenv()

    st.header("Chat with PDF")
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')  

    if pdf is not None:
        text = get_pdf_text(pdf)
        chunks = get_text_chunks(text)
        vector_store = get_vector_store(chunks, pdf.name)
        chain = get_conversational_chain()

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = vector_store.similarity_search(query=query)

            response = chain(
                {"input_documents":docs,
                 "question": query,
                 "return_only_outputs": True
                 }
            )
            #print(response)
            st.write_stream(stream_data(response["output_text"]))

if __name__ == '__main__':
    main()