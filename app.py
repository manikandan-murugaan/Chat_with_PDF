import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("AIzaSyAN2gnyYJaU0dZ7_xr89rWaSFpN5i023bA")
genai.configure(api_key=os.getenv("AIzaSyAN2gnyYJaU0dZ7_xr89rWaSFpN5i023bA"))

def get_pdf_text(pdf):
    txt=""
    for pdf in pdf:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            txt+=page.extract_text()
    return txt

def get_text_chunks(txt):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(txt)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAN2gnyYJaU0dZ7_xr89rWaSFpN5i023bA")
    vector_store=FAISS.from_texts(chunks,embedding=embeddings)
    vector_store.save_local("faiss.index")

def get_conversational_chain():
    genai.configure(api_key="AIzaSyAN2gnyYJaU0dZ7_xr89rWaSFpN5i023bA")
    prompt_template="""
    Answer the question as detailed as Possible from peovided context make sure to provide all details,if the answer is not available
    in the provided context just say,"answer is not available in the context",don't provide the wrong answer\n\n
    context:\n{context}?\n
    question: \n{question}\n
    
    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro")

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(question):
    embeddings=GoogleGenerativeAIEmbeddings(GOOGLE_API_KEY="AIzaSyAN2gnyYJaU0dZ7_xr89rWaSFpN5i023bA",model="models/embedding-001")

    new_db=FAISS.load_local("faiss.index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(question)
    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs,"question":question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply ",response["output_text"])

def main():
    st.set_page_config("chat pdf")
    st.header("chat pdf")
    question=st.text_input("ask question which related to pdf")

    with st.sidebar:
        st.title("Menu")
        pdf=st.file_uploader("Upload Your PDF",accept_multiple_files=True)
        if st.button("Submit & process"):
            with st.spinner("Procrssing"):
                raw_text=get_pdf_text(pdf)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    if question:
        user_input(question)

if __name__=="__main__":
    main()