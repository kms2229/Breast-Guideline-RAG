import os
import json
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.runnables import RunnablePassthrough
from RAPTOR import *
from AnyFile_Loader import *
from langchain.load import dumps, loads
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# LangSmith API key
os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_PROJECT']= st.secrets['LANGCHAIN_PROJECT']

os.environ['OPENAI_API_KEY'] = st.session_state.api_key_final
embd = st.session_state.embedding
model = st.session_state.model

file_path = st.session_state.file_path

def get_full_papers(docs):
    """Get full paper text from JSON file based on document IDs"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    paper_ids = set([doc.metadata.get('paper_id', doc.metadata.get('source', '')) for doc in docs])
    return [data[pid] for pid in paper_ids if pid in data]

# Multi Query: Different Perspectives
template_qd = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (5 queries):
Just give me the question and not your thought process.
"""
prompt_decomposition = ChatPromptTemplate.from_template(template_qd)

generate_queries = (
    prompt_decomposition 
    | model 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(docs: list[list]):
    """Unique union of retrieved docs"""
    flattened_docs = [dumps(doc) for sublist in docs for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def setup_language_model_chain(vectorstore, topk: int):
    print(">>>Setting up LLM chain...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": topk})
    print(f"Top-K: {topk}")

    retrieval_chain = generate_queries | model| StrOutputParser()

    template = """
                    Previous Conversation:
                    {chat_history}

                    Answer the question comprehensively and with detailed logical points based on the following context and previous conversation:
                    {context}

                    Question: {question}

                    If the answer to the question is not present in the given context or previous conversation, respond with "I don't have enough information to answer this question."
                    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        with open(file_path, 'r') as file:
            data = json.load(file)
        paper_ids = set([doc.metadata.get('paper_id', doc.metadata.get('source', '')) for doc in docs])
        return "\n\n".join(data[pid] for pid in paper_ids if pid in data)

    rag_chain = (
        {
            "chat_history": RunnablePassthrough(),
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    print(">>>Chain setup completed.")
    print("="*30)
    return rag_chain

def invoke_chain(chain, question, context):
    print("Invoking the RAG chain...")
    try:
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                chat_history.append(f"Assistant: {msg['content']}")
        chat_history_str = "\n".join(chat_history)
        
        response = chain.stream({
            "question": question,
            "context": context,
            "chat_history": chat_history_str
        })
        response = list(response)
        return response
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        return "Error processing your request."


def load_vectorstore_pinecone(index_name: str, embedding_function) -> PineconeVectorStore:
    print(f">>>Loading Pinecone vectorstore: {index_name}")
    
    # Get Pinecone API key from secrets
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in secrets.toml")
    
    # Initialize Pinecone
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    
    # Load from existing index
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_function
    )
    
    print(">>>Pinecone Vectorstore loaded.")
    print("="*30)
    return vectorstore
