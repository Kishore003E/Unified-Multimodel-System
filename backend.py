# backend.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
from gtts import gTTS
import os

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
        return f"Error: {e}"

def save_audio(text, file_name="final_answer.mp3"):
    tts = gTTS(text)
    tts.save(file_name)
    return file_name

#AudioProcessing
# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
def transcribe_audio(audio_file_path, chunk_length_s=30):
    # Load the audio file
    audio, sampling_rate = torchaudio.load(audio_file_path)
    
    # If the audio file has multiple channels, convert it to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0).unsqueeze(0)
    
    # Resample the audio to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    audio = resampler(audio).squeeze()
    
    # Calculate the number of samples in each chunk
    chunk_length_samples = chunk_length_s * 16000  # 16kHz sampling rate
    
    # Split the audio into chunks
    audio_chunks = audio.split(chunk_length_samples)
    
    # Process each chunk
    full_transcription = ""
    for i, chunk in enumerate(audio_chunks):
        print(f"Processing chunk {i + 1}/{len(audio_chunks)}")
        
        # Convert chunk to the required format
        input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features

        # Generate token ids
        predicted_ids = model.generate(input_features)

        # Decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # Append the transcription
        full_transcription += transcription[0] + " "

    return full_transcription.strip()

# Test the function
def speech_recognizer(path):
    audio_file_path = path
    transcript = transcribe_audio(audio_file_path)
    print(transcript)
    return transcript


