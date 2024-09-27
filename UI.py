# app.py
import streamlit as st
from backend import load_pdf, chunking, vectordb, queryretriver, generator, save_audio, speech_recognizer
from Clipmodel import image_analyzer
st.title("</> File Analyzer using RAG-arc")

# Select file type
 
file_type = st.radio("Select the file type", ("PDF", "Image", "Audio"))

# File uploader
uploaded_file = None
if file_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
elif file_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image file", type=["png", "jpg", "jpeg"])
elif file_type == "Audio":
    uploaded_file = st.file_uploader("Upload an Audio file", type=["wav", "mp3", "flac", "ogg", "aiff", "aac", "m4a"])

if uploaded_file is not None:
    if file_type == "PDF":
        with st.spinner("Loading and processing PDF..."):
            temp_pdf_path = "uploaded_file.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process PDF
            pdf_data, num_pages, total_words = load_pdf(temp_pdf_path)
            chunked_docs, num_chunks = chunking(pdf_data)
            database = vectordb(chunked_docs)

            st.success("PDF loaded and processed successfully!")
            st.write(f"Number of Pages: **{num_pages}**")
            st.write(f"Total Number of Words: **{total_words}**")
            st.write(f"Number of Chunks: **{num_chunks}**")
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

            question = st.text_input("Enter your query: ")
            if st.button("Generate Answer"):
                with st.spinner("Generating answer..."):
                    related_docs = queryretriver(question, database)
                    final_answer = generator(related_docs, question)
                    if final_answer:
                        st.subheader("Generated Answer")
                        st.write(final_answer)
                        st.code(final_answer, language="text")
                        audio_file = save_audio(final_answer)
                        st.audio(audio_file)
    elif file_type == "Image":
        with st.spinner("Processing Image..."):
                # Save the uploaded image temporarily
            temp_image_path = "uploaded_image_file.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.read())
                
            st.success(" Image processed successfully!")
            query = st.text_input("Enter your query about the image:")
            if st.button("Generate answer"):
                with st.spinner("Generating answer..."):
                    image_answer = image_analyzer(temp_image_path, query)
                
                    st.success("Image processed successfully!")
                    st.write(f"Generated Answer: **{image_answer}**")
        
                
                    
                
                # Perform image analysis
                


    elif file_type == "Audio":
        with st.spinner("Transcribing audio..."):
            temp_audio_path = "uploaded_audio_file.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            # Transcribe audio
            transcription = speech_recognizer(temp_audio_path)
            st.success("Audio transcribed successfully!")
            st.write(f"Transcription: **{transcription}**")

            # Process transcription as text
            chunked_docs, num_chunks = chunking([transcription])
            database = vectordb(chunked_docs)

            question = st.text_input("Enter your query: ")
            if st.button("Generate Answer"):
                with st.spinner("Generating answer..."):
                    related_docs = queryretriver(question, database)
                    final_answer = generator(related_docs, question)
                    if final_answer:
                        st.subheader("Generated Answer")
                        st.write(final_answer)
                        st.code(final_answer, language="text")
                        audio_file = save_audio(final_answer)
                        st.audio(audio_file)

