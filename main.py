import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from typing import Generator
import time
import random
import string


def load_split_document(pdf_path):
    """Load a PDF document and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_documents(data)
    return chunks

def index_chunks(chunks, vector_store):
    """Index the documents in Pinecone."""
    vector_store.add_documents(documents=chunks)


def convo_chain():
    prompt_tmpl = """
    Answer the question in as much detail as possible, but within the context provided. Do not invent information that is not in the context provided.
    If you cannot find related information, simply respond: "We were unable to find related information. Please try another approach." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer it in the same language as the context.
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1.0)
    prompt = PromptTemplate(template=prompt_tmpl, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def answer_stream(question, vector_store) -> Generator[str, None, None]:
    answer = "I'm sorry, I can't answer that question."
    
    # Search for the closest document
    closest_docs = vector_store.similarity_search(question, k=3)
    if closest_docs:
        context = [doc.page_content for doc in closest_docs]
        chain = convo_chain()
        answer = chain({"input_documents": closest_docs, "context": context, "question": question}, return_only_outputs=True)["output_text"]
    
    for word in answer.split():
        yield word + " "
        time.sleep(0.1)
    

def main():
    load_dotenv()
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    INDEX_NAME = "topicos3-pdf-assistant"
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=INDEX_NAME, metric="cosine", dimension=768, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    
    index = pc.Index(INDEX_NAME)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Graphical User Interface
    st.title("Talk to PDF")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        
    chat = st.container()
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    # PDF Upload
    st.sidebar.title("PDF Upload")
    st.sidebar.header("Upload your PDF files")
    st.session_state.uploaded_files = st.sidebar.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)
    left, right = st.sidebar.columns(2)
    submit_button = left.button("Submit", use_container_width=True, type="primary")
    reset_button = right.button("Reset", use_container_width=True)
    if submit_button:
        # Load and split the PDF documents
        with st.spinner("Processing documents..."):
            for pdf_file in st.session_state.uploaded_files:
                # Save the PDF file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(pdf_file.read())
                    temp_file_path = temp_file.name

                # Load and split the document
                index_chunks(load_split_document(temp_file_path), vector_store)

                # Remove the temporary file
                os.remove(temp_file_path)
        st.success("Documents processed successfully")
    elif reset_button:
        st.session_state.clear()
        
    if "uploaded_files" not in st.session_state or not st.session_state.uploaded_files:
        st.warning("Please upload a PDF file")
        st.stop()
    else:
        if input := st.chat_input("Ask me anything"):
            chat.chat_message("user").write(input)
            st.session_state.messages.append({"role": "user", "content": input})
            
            answer = chat.chat_message("assistant").write_stream(answer_stream(input, vector_store))
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
