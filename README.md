# Breast Guideline RAG System

An Advanced Retrieval-Augmented Generation (RAG) system for querying breast cancer treatment guidelines and research papers.

## Features

- **Query Decomposition**: Breaks complex questions into 5 sub-questions
- **RAPTOR Clustering**: Hierarchical document organization
- **Multiple LLM Support**: OpenAI (GPT-3.5, GPT-4, GPT-4o, o1) and DeepSeek
- **Conversation History**: Maintains context across multiple turns
- **Pinecone Cloud Storage**: Scalable cloud-based vector database
- **Streaming Responses**: Real-time answer generation

## Dataset

- **Source**: 500+ breast cancer guideline PDFs
- **Location**: `/Users/kavachshah/Downloads/degrees/PROJ__NLP/RA/Breast Guideline PDFs`
- **Topics**: Breast reconstruction, mastectomy, surgical techniques, oncology guidelines

## Setup Instructions

### 1. Create Virtual Environment

```bash
cd /Users/kavachshah/Downloads/degrees/PROJ__NLP/RA/Breast-Guideline-RAG

# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Get Pinecone Account

1. Sign up at https://www.pinecone.io/
2. Get your API key from dashboard
3. See `PINECONE_SETUP.md` for detailed instructions

### 4. Configure API Keys

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and edit:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
PINECONE_API_KEY = "pcsk-your-pinecone-api-key-here"
PINECONE_INDEX_NAME = "breast-guidelines"
LANGCHAIN_API_KEY = ""  # Optional
LANGCHAIN_PROJECT = "Breast-Guideline-RAG"
FOLDER_PATH = "./Dataset"
PDF_SOURCE_PATH = "/Users/kavachshah/Downloads/degrees/PROJ__NLP/RA/Breast Guideline PDFs"
```

### 5. Build Vector Store

**IMPORTANT**: Run this first to process PDFs and upload to Pinecone:

```bash
python build_vectorstore.py
```

This will:
- Load all 500+ PDFs
- Split into chunks
- Create embeddings (OpenAI)
- Create Pinecone index (if doesn't exist)
- Upload all vectors to Pinecone cloud
- Generate full-text JSON file

**Expected time**: 15-30 minutes  
**Cost**: ~$2-5 (OpenAI embeddings)

### 6. Run the Application

```bash
streamlit run Chat_UI.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Enter API Key**: In the sidebar, enter your OpenAI or DeepSeek API key
2. **Select Model**: Choose from available LLM models
3. **Set Top-K**: Adjust number of documents to retrieve (default: 3)
4. **Ask Questions**: Type your question about breast cancer guidelines

### Example Questions

- "What are the current guidelines for nipple-sparing mastectomy?"
- "What are the complications of prepectoral breast reconstruction?"
- "How does radiation therapy affect breast reconstruction outcomes?"
- "What are the indications for acellular dermal matrix in reconstruction?"
- "Compare DIEP flap vs implant-based reconstruction"

## System Architecture

```
User Question
    ↓
Query Decomposition (5 sub-questions)
    ↓
Vector Store Retrieval (Top-K per sub-question)
    ↓
Full Paper Retrieval (from JSON)
    ↓
Sub-Answer Generation (with chat history)
    ↓
Final Answer Synthesis
    ↓
Streaming Response
```

## File Structure

```
Breast-Guideline-RAG/
├── Chat_UI.py                      # Main Streamlit app
├── Query_Decomposition_RAG.py      # RAG pipeline
├── RAPTOR.py                       # Hierarchical clustering
├── AnyFile_Loader.py               # Document loader
├── build_vectorstore.py            # Pinecone upload script
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── PINECONE_SETUP.md               # Pinecone setup guide
├── .streamlit/
│   ├── secrets.toml                # API keys & config
│   └── secrets.toml.example        # Template
└── Dataset/
    └── breast_guidelines_full_text.json  # Full paper texts
```

## Troubleshooting

### Issue: "PINECONE_API_KEY not found"
Add your Pinecone API key to `.streamlit/secrets.toml`

### Issue: "Index not found"
Run `python build_vectorstore.py` to create and upload to Pinecone.

### Issue: "API rate limit exceeded"
- Reduce Top-K value
- Use GPT-3.5-turbo instead of GPT-4
- Wait a few minutes between queries

### Issue: "Out of memory"
- Reduce chunk size in `build_vectorstore.py`
- Reduce Top-K retrieval value
- Process PDFs in batches

## Performance & Costs

### Query Performance
- **Query Time**: ~15-20 seconds per question
- **Cost per Query** (GPT-4): ~$0.06
- **Cost per Query** (GPT-3.5): ~$0.01

### Monthly Costs
- **Pinecone**: $70/month (Starter plan)
- **OpenAI** (100 queries/month): $1-6
- **Total**: ~$75/month

### One-Time Setup
- **Embeddings**: $2-5 (500 PDFs)

## Advanced Features

### Conversation History

The system maintains conversation context:

```
User: What is nipple-sparing mastectomy?
Assistant: [Detailed answer]

User: What are the complications?
Assistant: [Answer considering previous context about NSM]
```

### Expandable UI Sections

- **Generated Questions**: See the 5 sub-questions
- **Retrieved Documents**: View which papers were used
- **Final Context**: See the Q&A pairs fed to final answer
- **Conversation History**: Full chat log

### Multiple LLM Support

Switch between models based on needs:
- **GPT-3.5-turbo**: Fast, cheap
- **GPT-4**: High quality
- **GPT-4o-mini**: Balanced
- **o1-mini**: Advanced reasoning
- **DeepSeek**: Alternative provider

## Citation

If you use this system in research, please cite the original RAPTOR paper and LangChain framework.

## License

This system is for research and educational purposes.

## Support

For issues or questions, refer to the original Advanced-RAG system documentation.
