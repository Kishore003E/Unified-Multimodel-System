from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import chromadb
from chromadb.config import Settings
import numpy as np
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption




def image_embed(model, processor, image_path):
    try:
        image = Image.open(image_path)
        
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.get_image_features(**inputs)
        
        image_embeddings = outputs.detach().numpy().flatten().tolist()
        return image_embeddings
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def image_vectordb(image_embeddings, metadata):
    client = chromadb.Client(Settings())
    
    collection_name = "image_embeddings"
    existing_collections = [col.name for col in client.list_collections()]
    
    if collection_name in existing_collections:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(name=collection_name)
    
    collection.add(
        ids=["img_0"],  
        embeddings=[image_embeddings],
        metadatas=[metadata]
    )
    
    return collection

def classify_image(model, processor, image_path, candidate_labels):
    image = Image.open(image_path)
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)
    best_label = candidate_labels[probs.argmax()]
    return best_label


def image_queryretriver(query_embedding, collection):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  
    )
    return results["metadatas"]

def image_generator(metadata_list, question):
    context_text = "\n".join(str(metadata) for metadata in metadata_list)
    messages = [("human", f"Question: {question}\nContext: {context_text}\nAnswer:")]
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="AIzaSyB8LUuwLtTnlFFQQwZSgr-JdMrNE2h2VsE")
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        return f"Error: {e}"

def image_analyzer(image_path,query):
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
    image_embeddings = image_embed(model, processor, image_path)
    if image_embeddings:
        
        candidate_labels = ["a cat", "a mountain", "a painting", "a cityscape", "a fantasy scene", "nature", "sci-fi", "anime"]
        #description = classify_image(model, processor, image_path, candidate_labels)
        description = generate_caption(image_path)
        metadata = {"image_path": image_path, "description": description}
        

        
        collection = image_vectordb(image_embeddings, metadata)
        
        metadata_list = image_queryretriver(image_embeddings, collection)
        
        question = query
        answer = image_generator(metadata_list, question)
        return answer
