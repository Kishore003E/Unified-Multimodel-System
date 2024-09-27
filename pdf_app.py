import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from gtts import gTTS


# Backend functions
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    page_texts = [page.page_content for page in pages]
    num_pages = len(pages)
    total_words = sum(len(page.page_content.split()) for page in pages)
    return page_texts, num_pages, total_words

def chunking(page_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents(page_texts)
    num_chunks = len(docs)
    return docs, num_chunks

def vectordb(docs):
    # Initialize the HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Chroma vector database using the HuggingFace embeddings
    db = Chroma.from_documents(docs, embeddings)
    return db

def queryretriver(question, db):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="...")
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    unique_docs = retriever_from_llm.invoke(question)
    return unique_docs

def generator(unique_docs, question):
    context_text = "\n".join(doc.page_content for doc in unique_docs)
    messages = [("human", f"Question: {question}\nContext: {context_text}\nAnswer:")]
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="...")
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit UI
st.title("📜 PDF Analyzer using RAG-arc")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Loading and processing PDF..."):
        # Save uploaded file temporarily
        temp_pdf_path = "uploaded_file.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load and process the PDF
        pdf_data, num_pages, total_words = load_pdf(temp_pdf_path)
        chunked_docs, num_chunks = chunking(pdf_data)
        database = vectordb(chunked_docs)

        st.success("PDF loaded and processed successfully!")

        # Display PDF info
        st.write(f"Number of Pages: *{num_pages}*")
        st.write(f"Total Number of Words: *{total_words}*")
        st.write(f"Number of Chunks: *{num_chunks}*")

        # Custom CSS for underlined text
        st.markdown("""
        <style>
        .underline-color {
            text-decoration: underline;
            text-decoration-color: white;
            font-size: 25px;
        }
        .highlight {
            background-color: #262730;
            padding: 5px;
            border-radius: 6px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Display underlined and highlighted text
        st.markdown('<p class="underline-color">You can also use the following Queries...</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">Generate a concise summary of the document.</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">Provide a TL;DR for this document.</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">What are the main points of this document?</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">Extract the key arguments presented in the document</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">Identify the sections and subsections of this document.</p>', unsafe_allow_html=True)

        # Query input
        question = st.text_input("Enter your query: ")

        if st.button("Generate Answer"):
            with st.spinner("Generating answer..."):
                related_docs = queryretriver(question, database)
                final_answer = generator(related_docs, question)
                
                if final_answer:
                    st.subheader("Generated Answer")
                    st.write(final_answer)

                    st.code(final_answer, language="text")
                                        
                    if final_answer:
                        tts = gTTS(final_answer)
                        tts.save("final_answer.mp3")
                        
                        st.audio("final_answer.mp3")
