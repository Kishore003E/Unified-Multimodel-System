from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from langchain_google_genai import ChatGoogleGenerativeAI


# Load model and processor


def image_embed(model, processor, image_path):
    try:
        # Load the image using PIL
        image = Image.open(image_path)
        
        # Process the image and extract features
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.get_image_features(**inputs)
        
        # Convert outputs to a flat list
        image_embeddings = outputs.detach().numpy().flatten().tolist()
        return image_embeddings
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def image_vectordb(image_embeddings, metadata):
    # Initialize ChromaDB client
    client = chromadb.Client(Settings())
    
    # Create or get the collection
    collection_name = "image_embeddings"
    existing_collections = [col.name for col in client.list_collections()]
    
    if collection_name in existing_collections:
        # If the collection already exists, get it
        collection = client.get_collection(collection_name)
    else:
        # Otherwise, create a new collection
        collection = client.create_collection(name=collection_name)
    
    # Add the embedding and metadata to the collection
    collection.add(
        ids=["img_0"],  # Assuming single image for simplicity
        embeddings=[image_embeddings],
        metadatas=[metadata]
    )
    
    return collection

def image_queryretriver(query_embedding, collection):
    # Query the collection using the image embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Adjust the number of results to your needs
    )
    return results["metadatas"]

def image_generator(metadata_list, question):
    context_text = "\n".join(str(metadata) for metadata in metadata_list)
    messages = [("human", f"Question: {question}\nContext: {context_text}\nAnswer:")]
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="...")
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        return f"Error: {e}"

# Example usage



def image_analyzer(image_path,query):
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    image_embeddings = image_embed(model, processor, image_path)
    if image_embeddings:
        metadata = {"image_path": image_path}
        
        # Store embeddings
        collection = image_vectordb(image_embeddings, metadata)
        
        # Query example
        metadata_list = image_queryretriver(image_embeddings, collection)
        
        # Generate answer
        question = query
        answer = image_generator(metadata_list, question)
        return answer

#image_path = "/home/kamalganth/VSCode/Audio_analyzer/WhatsApp Image 2024-08-13 at 3.57.34 PM.jpeg"
#query = "What the image says"
#final_output =image_analyzer(image_path,query)

#print(final_output)