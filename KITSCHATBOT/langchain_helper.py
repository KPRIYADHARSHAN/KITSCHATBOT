import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
)

prompt_template = """
You are KITSChatBot, an assistant for Karunya Institute of Technology and Sciences. Use the following context to answer the question accurately and concisely. If the answer isn’t in the context, say "I don’t have enough information to answer this. Please contact admissions@karunya.edu."

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def create_vector_db():
    if not os.path.exists("KITS_FAQ.csv"):
        raise FileNotFoundError("KITS_FAQ.csv not found in the project directory. Please add it and try again.")
    df = pd.read_csv("KITS_FAQ.csv")
    documents = [
        f"Section: {row['Section']}\nQuestion: {row['Question']}\nAnswer: {row['Answer']}"
        for _, row in df.iterrows()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(documents, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain