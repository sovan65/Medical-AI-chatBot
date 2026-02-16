import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CONTEXT_CHARS = 6000
SIMILARITY_SCORE_THRESHOLD = 0.85

# Function to read text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    # Smaller chunks improve retrieval precision and reduce irrelevant context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]

# Function to create and save a vector store
def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available to index.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
You are a medical document assistant. Use ONLY the provided context.
If the answer is not in the context, reply exactly: "The answer is not available in the context."
Keep the answer concise (3-6 sentences) unless the user asks for more detail.

Context:\n{context}\n
Question:\n{question}\n
Answer:
"""
    model = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0.2,
        max_output_tokens=512,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return model, prompt

# Function to handle user input and generate a response
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Use MMR to reduce redundant chunks, then filter by similarity score.
    mmr_docs = new_db.max_marginal_relevance_search(user_question, k=6, fetch_k=20)
    scored_docs = new_db.similarity_search_with_score(user_question, k=6)

    scored_map = {doc.page_content: score for doc, score in scored_docs}
    filtered_docs = [doc for doc in mmr_docs if scored_map.get(doc.page_content, 1.0) <= SIMILARITY_SCORE_THRESHOLD]

    if not filtered_docs:
        return "The answer is not available in the context."

    # Build a context string and run the prompt directly through the model
    model, prompt = get_conversational_chain()
    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    context = context[:MAX_CONTEXT_CHARS]
    prompt_text = prompt.format(context=context, question=user_question)
    response = model.invoke(prompt_text)
    response_text = None
    if hasattr(response, "content"):
        response_text = response.content
    elif hasattr(response, "text"):
        response_text = response.text
    elif isinstance(response, dict):
        response_text = response.get("content") or response.get("text") or response.get("output")
    elif isinstance(response, list) and response:
        first = response[0]
        response_text = getattr(first, "content", None) or getattr(first, "text", None) or str(first)
    elif isinstance(response, str):
        response_text = response

    if not response_text:
        return "The answer is not available in the context."

    if isinstance(response_text, list):
        parts = []
        for item in response_text:
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or item.get("output")
                if text_value:
                    parts.append(str(text_value))
            elif item:
                parts.append(str(item))
        response_text = "\n".join(parts)

    # Strip any leaked metadata from the model response.
    response_text = re.sub(r"\s*[,{]\s*['\"]extras['\"].*", "", response_text, flags=re.S).rstrip()

    return response_text

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon=":robot_face:")

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .st-emotion-cache-1y4p8pa {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-16txtl3 {
        font-family: 'Arial', sans-serif;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        border-radius: 10px;
    }
    .st-emotion-cache-1kyxreq.e115fcil2 {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for uploading files
    with st.sidebar:
        st.title("Menu")
        st.image("https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", width="stretch")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.warning("Please upload at least one PDF.")
                    return
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No extractable text found in the uploaded PDF(s).")
                    return
                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.error("No usable text chunks were created from the PDF(s).")
                    return
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area
    st.title("Medical Chatbot")
    st.header("Ask questions about your medical documents")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
