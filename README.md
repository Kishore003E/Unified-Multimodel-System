# Unified Multimodal System 🤖🌐

## Overview

The **Unified Multimodal System** is a cutting-edge **Gen-AI** framework that leverages the **Retrieval-Augmented Generation (RAG)** architecture to efficiently process and analyze multiple data modalities such as **PDFs**, **audio**, and **images**. This system integrates various modern technologies to enable seamless multimodal retrieval and processing. 🚀

With features like **audio generation**, **text-to-speech**, and **intelligent multimodal retrieval**, the system provides powerful AI-driven solutions for various applications. 🔍🎧📄

## Features

- **Multimodal Processing** 🧠:
  - Supports the analysis and retrieval of information from **PDFs**, **audio**, and **images**. 🗂️🎧📸
  - Efficiently processes data through intelligent integration of AI models and retrieval techniques.

- **Efficient Retrieval and Analysis** ⚙️:
  - Utilizes **LangChain** for orchestrating workflows and interacting with different data sources.
  - **Sentence-Transformers** and **Chroma DB** power the semantic search and retrieval process. 🔎
  - **MultiQuery-Retriever** improves the efficiency of data retrieval from multiple sources simultaneously. 🛠️

- **Audio Generation and Text-to-Speech** 🎤:
  - Integrated with **Gemini** and **GTTS** (Google Text-to-Speech) for generating high-quality speech from text. 🗣️

- **Streamlit Interface** 🌐:
  - A user-friendly web interface built using **Streamlit** that allows easy interaction with the system.

## Tech Stack 🛠️

- **Programming Language**: Python 🐍

- **Key Libraries & Frameworks**:
  - **LangChain**: For chaining language models and creating advanced workflows.
  - **Chroma DB**: A vector database for semantic search and indexing. 🔥
  - **GTTS (Google Text-to-Speech)**: For converting text into natural-sounding speech.
  - **HuggingFace**: For transformer-based models and APIs.
  - **Streamlit**: For creating the frontend interface.

## Installation 💻

### Prerequisites
Make sure you have **Python 3.x** installed.

1. Clone this repository:

```
git clone https://github.com/yourusername/Unified-Multimodal-System.git
cd Unified-Multimodal-System
```

2. Install the required dependencies:

``` 
pip install -r requirements.txt
```

## Running the System 🚀
- To launch the Streamlit interface, run the following command:
```
streamlit run UI.py
```

## How It Works 🤖🔍
- Data Ingestion: Upload your PDF, audio, or image to the system.

- Multimodal Analysis: The system processes the data using integrated models for retrieval and generation.

- Results: The processed information is displayed or spoken (for audio) through the Streamlit interface.

## Contributing 🤝
We welcome contributions! If you'd like to contribute to the project, feel free to open an issue or submit a pull request. Make sure to fork the repo and follow the contributing guidelines.
