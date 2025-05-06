import streamlit as st
from backend import load_pdf, chunking, vectordb, queryretriver, generator, save_audio, speech_recognizer
from Clipmodel import image_analyzer

# --- PAGE CONFIG ---
st.set_page_config(page_title="File Analyzer using RAG", layout="wide")
st.title("📒 File Analyzer using RAG")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    body {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .reportview-container {
        background: #0E1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00CED1;
    }
    .stButton>button {
        background-color: #00CED1;
        color: white;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
    }
    .stTextInput>div>input {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("📂 Upload Your File")
file_type = st.sidebar.radio("Select File Type", ("PDF", "Image", "Audio"))

# --- TABS BASED ON FILE TYPE ---
tab1, tab2, tab3 = st.tabs(["📄 PDF", "🖼️ Image", "🎧 Audio"])

# === PDF TAB ===
with tab1:
    if file_type == "PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                temp_pdf_path = "uploaded_file.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                pdf_data, num_pages, total_words = load_pdf(temp_pdf_path)
                chunked_docs, num_chunks = chunking(pdf_data)
                database = vectordb(chunked_docs)

                st.success("PDF loaded and processed!")
                with st.expander("📊 File Summary"):
                    st.write(f"**Pages:** {num_pages}")
                    st.write(f"**Total Words:** {total_words}")
                    st.write(f"**Chunks:** {num_chunks}")

                st.markdown("#### 💡 Suggested Queries")
                st.markdown("""
                - Generate a concise summary of the document.  
                - Provide a TL;DR for this document.  
                - What are the main points of this document?  
                - Extract the key arguments presented.  
                - Identify the sections and subsections.  
                """)

                query = st.text_input("🔍 Enter your query")
                if st.button("Generate Answer", key="pdf_btn"):
                    with st.spinner("Generating answer..."):
                        related_docs = queryretriver(query, database)
                        final_answer = generator(related_docs, query)
                        if final_answer:
                            st.subheader("📝 Answer")
                            st.write(final_answer)
                            st.code(final_answer, language="text")
                            audio_file = save_audio(final_answer)
                            st.audio(audio_file)

# === IMAGE TAB ===
with tab2:
    if file_type == "Image":
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            temp_image_path = "uploaded_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.read())

            st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
            query = st.text_input("🔍 Query about the image")
            if st.button("Generate Answer", key="img_btn"):
                with st.spinner("Analyzing Image..."):
                    image_answer = image_analyzer(temp_image_path, query)
                    st.success("Answer Generated!")
                    st.subheader("📝 Answer")
                    st.write(image_answer)

# === AUDIO TAB ===
with tab3:
    if file_type == "Audio":
        uploaded_file = st.sidebar.file_uploader("Upload Audio", type=["wav", "mp3", "flac", "ogg", "aiff", "aac", "m4a"])
        if uploaded_file:
            temp_audio_path = "uploaded_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Transcribing Audio..."):
                transcription = speech_recognizer(temp_audio_path)
                st.success("Transcription Complete!")
                st.write(f"🗒️ Transcription: **{transcription}**")

                chunked_docs, num_chunks = chunking([transcription])
                database = vectordb(chunked_docs)

                query = st.text_input("🔍 Enter your query")
                if st.button("Generate Answer", key="audio_btn"):
                    with st.spinner("Generating answer..."):
                        related_docs = queryretriver(query, database)
                        final_answer = generator(related_docs, query)
                        if final_answer:
                            st.subheader("📝 Answer")
                            st.write(final_answer)
                            st.code(final_answer, language="text")
                            audio_file = save_audio(final_answer)
                            st.audio(audio_file)
