import streamlit as st
st.set_page_config(page_title="Breast Guideline Query Interface")
import os
import io
import sys
import requests
import openai
from openai import OpenAI
from langchain_openai import OpenAI,OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv

#################################################################################################################    
##################################   Util Functions   ###########################################################
#################################################################################################################
load_dotenv()

class CapturePrints:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.captured_output = io.StringIO()

    def __enter__(self):
        sys.stdout = self.captured_output
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = sys.__stdout__
        if self.log_callback:
            self.log_callback(self.captured_output.getvalue())

# Initialize session state
if 'log' not in st.session_state:
    st.session_state.log = ""

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def update_log(message):
    st.session_state.log += message

def validate_openai_api_key(api_key: str) -> bool:
    """Validates the OpenAI API key"""
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.APIConnectionError:
        return False
    except Exception as e:
        return False
    else:
        return True
    
def validate_deepseek_api_key(api_key: str) -> bool:
    """Validates a DeepSeek API key"""
    url = "https://api.deepseek.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code // 100 == 2:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return False
    
with st.sidebar:
    st.title('Breast Guideline Query Interface')
    
    model_selected = st.selectbox(
        "Select the Model",
        ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo","gpt-4o-mini","gpt-4o","o1-mini","o1","deepseek-chat","deepseek-reasoner"),
        index=None,
        placeholder="Select the Model",
    )

    api_key = st.sidebar.text_input('Enter API key for the AI Model (DeepSeek/OpenAI):', type='password', key='api_key')
    if not api_key:
        st.stop()

    if api_key and model_selected:
        if model_selected in ("deepseek-chat","deepseek-reasoner"):
            api_key_valid = validate_deepseek_api_key(api_key) 
            embd_key = st.sidebar.text_input('Enter OpenAI API key for the Embedding Model:', type='password', key='embd_key')
            if not embd_key:
                st.stop()
        else: 
            api_key_valid = validate_openai_api_key(api_key)
            embd_key = api_key
        
        if api_key_valid:
            st.sidebar.success('API key validated!', icon='✅')
            st.session_state.api_key_final = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            st.session_state.embedding = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=embd_key)
        else:
            if not validate_openai_api_key(embd_key):
                st.sidebar.error('Please enter a valid OpenAI API key for the embedding Model to continue.', icon='⚠️')
            else:
                if model_selected in ("deepseek-chat","deepseek-reasoner"):
                    st.sidebar.error('Please enter a valid DeepSeek API key for the AI Model to continue.', icon='⚠️')
                else:
                    st.sidebar.error('Please enter a valid OpenAI API key for the AI Model to continue.', icon='⚠️')
            st.stop()

    # Dataset path
    file_path = os.path.join(st.secrets["FOLDER_PATH"], "breast_guidelines_full_text.json")
    st.session_state.file_path = file_path
    st.session_state.model_selected = model_selected
    
    if model_selected in ("o1-mini","o1"):
        st.session_state.model = ChatOpenAI(temperature=1, model=st.session_state.model_selected)
    elif model_selected in ("deepseek-chat","deepseek-reasoner"):
        st.session_state.deepseek_client = OpenAI(
            api_key=st.session_state.api_key_final,
            base_url="https://api.deepseek.com/v1"
        )
        st.session_state.model = ChatOpenAI(
            temperature=0,
            model=st.session_state.model_selected,
            openai_api_key=st.session_state.api_key_final,
            openai_api_base="https://api.deepseek.com/v1"
        )
    else:
        st.session_state.model = ChatOpenAI(temperature=0, model=st.session_state.model_selected)
    
    model = st.session_state.model
    from Query_Decomposition_RAG import *
    
    # Pinecone index name
    pinecone_index = st.secrets.get("PINECONE_INDEX_NAME", "breast-guidelines")
    
    with CapturePrints(log_callback=update_log):
        if 'vectorstore' not in st.session_state: 
            st.session_state.vectorstore = load_vectorstore_pinecone(pinecone_index, st.session_state.embedding)
            st.success('Pinecone VectorStore is Loaded')

    st.markdown('---')
    st.text_area("Log", st.session_state.log, height=300)

#################################################################################################################    
##################################   Chat Interface    ##########################################################    
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with breast cancer guidelines today?"}]

def display_conversation_history():
    """Show full conversation history in expander"""
    with st.expander("Conversation History"):
        for msg in st.session_state.messages:
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg['content']}")
            st.markdown("---")

def display_generated_questions(questions):
    with st.expander("Generated Questions"):
        for i, q in enumerate(questions):
            st.write(f"{q}")

def display_retrieved_documents(retrieved_docs):
    with st.expander("Retrieved Documents"):
        for i, docs in enumerate(retrieved_docs):
            st.write(f"For Q{i+1}:")
            for doc in docs:
                content = doc.page_content.split('\n')[:1]
                st.write(' '.join(content))

def display_final_context(final_context):
    with st.expander("Final Context"):
        st.write("Context used for final answer generation:")
        st.write(final_context)

col1, col2= st.columns([1,1])
with col2:
    st.session_state.topk = st.number_input("Top K Retrieval", min_value=1, max_value=10, value=3, help='Number of chunks Retrieved')

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

disable_chat = not(api_key_valid)

display_conversation_history()

if not disable_chat:
    if og_question := st.chat_input("Enter your question:", disabled=disable_chat):
        st.session_state.messages.append({"role": "user", "content": og_question})
        with st.chat_message("user"):
            st.write(og_question)

    st.session_state.rag_chain = setup_language_model_chain(st.session_state.vectorstore, topk=st.session_state.topk)
    
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.topk})
                questions = generate_queries.invoke({"question": og_question})
                display_generated_questions(questions)
                
                template = """
                    Previous Conversation:
                    {chat_history}

                    Answer the question comprehensively and with detailed logical points based on the following context and previous conversation:
                    {context}

                    Question: {question}

                    If the answer to the question is not present in the given context or previous conversation, respond with "I don't have enough information to answer this question."
                    """
                prompt1 = ChatPromptTemplate.from_template(template)
                rag_results = []
                
                for sub_question in questions:
                    retrieved_docs = retriever.invoke(sub_question)
                    full_papers = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    chat_history_subq = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history_subq.append(f"User: {msg['content']}")
                        elif msg["role"] == "assistant":
                            chat_history_subq.append(f"Assistant: {msg['content']}")
                    chat_history_subq_str = "\n".join(chat_history_subq)
                    
                    answer = (prompt1 | model | StrOutputParser()).invoke({
                        "context": full_papers,
                        "question": sub_question,
                        "chat_history": chat_history_subq_str
                    })
                    rag_results.append(answer)

                context = format_qa_pairs(questions, rag_results)
                retrieved_documents = [retriever.invoke(q) for q in questions]
                display_retrieved_documents(retrieved_documents)
                display_final_context(context)
                
                response = invoke_chain(st.session_state.rag_chain, og_question, context)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
