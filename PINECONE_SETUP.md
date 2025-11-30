# Pinecone Cloud Setup Guide

## What is Pinecone?

Pinecone is a **cloud-based vector database** that stores your document embeddings in the cloud instead of locally. This means:
- ✅ No local storage needed
- ✅ Access from anywhere
- ✅ Scalable to millions of documents
- ✅ Managed service (no maintenance)
- ❌ Requires subscription (~$70/month)
- ❌ Needs internet connection

---

## Step 1: Create Pinecone Account

1. Go to https://www.pinecone.io/
2. Click "Sign Up" (free trial available)
3. Choose a plan:
   - **Starter**: $70/month (recommended)
   - **Enterprise**: Custom pricing

---

## Step 2: Get Your Pinecone API Key

1. Log into Pinecone dashboard
2. Go to **API Keys** section
3. Click **Create API Key**
4. Copy your API key (starts with `pcsk_...`)
5. Save it securely

---

## Step 3: Configure Your System

### Edit `.streamlit/secrets.toml`

Copy from `.streamlit/secrets.toml.example` and fill in:

```toml
OPENAI_API_KEY = "sk-your-openai-key-here"
PINECONE_API_KEY = "pcsk-your-pinecone-key-here"
PINECONE_INDEX_NAME = "breast-guidelines"
LANGCHAIN_API_KEY = ""  # Optional
LANGCHAIN_PROJECT = "Breast-Guideline-RAG"
FOLDER_PATH = "./Dataset"
PDF_SOURCE_PATH = "/Users/kavachshah/Downloads/degrees/PROJ__NLP/RA/Breast Guideline PDFs"
```

---

## Step 4: Build & Upload Vector Store

```bash
# Activate virtual environment
source venv/bin/activate

# Run the build script
python build_vectorstore.py
```

### What Happens:
1. Loads all 500+ PDFs
2. Splits into chunks
3. Creates embeddings (OpenAI)
4. **Creates Pinecone index** (if doesn't exist)
5. **Uploads all embeddings to Pinecone cloud**
6. Creates local JSON for full-text retrieval

**Time**: 15-30 minutes  
**Cost**: ~$2-5 (OpenAI embeddings)

---

## Step 5: Run the Application

```bash
streamlit run Chat_UI.py
```

The app will:
1. Connect to Pinecone
2. Load the `breast-guidelines` index
3. Ready to answer questions!

---

## Pinecone Dashboard

You can monitor your index at https://app.pinecone.io/

**What you'll see:**
- Index name: `breast-guidelines`
- Dimension: 1536 (OpenAI embedding size)
- Metric: cosine
- Vector count: ~15,000-20,000 (depending on PDFs)
- Region: us-east-1 (AWS)

---

## Cost Breakdown

### One-Time Setup
- **OpenAI Embeddings**: $2-5 (500 PDFs)
- **Pinecone**: Free during setup

### Monthly Costs
- **Pinecone Starter**: $70/month
  - Includes: 1 index, 100K vectors, unlimited queries
- **OpenAI API** (per query): $0.01-0.06
  - Depends on model (GPT-3.5 vs GPT-4)

### Total Monthly Cost
- **Pinecone**: $70
- **OpenAI** (100 queries): $1-6
- **Total**: ~$75/month

---

## Advantages vs Local ChromaDB

| Feature | Pinecone Cloud | Local ChromaDB |
|---------|---------------|----------------|
| **Storage** | Cloud (unlimited) | Local disk (~500MB) |
| **Access** | From anywhere | Only this machine |
| **Speed** | Fast (optimized) | Faster (no network) |
| **Scalability** | Millions of vectors | Limited by disk |
| **Cost** | $70/month | Free |
| **Maintenance** | Managed | You manage |
| **Internet** | Required | Not required |

---

## Troubleshooting

### "Invalid API key"
- Check your Pinecone API key in secrets.toml
- Make sure it starts with `pcsk_`
- Verify it's active in Pinecone dashboard

### "Index not found"
- Run `python build_vectorstore.py` first
- Check index name matches in secrets.toml
- Verify index exists in Pinecone dashboard

### "Rate limit exceeded"
- Pinecone Starter allows unlimited queries
- Check OpenAI rate limits instead
- Wait 1 minute and retry

### "Connection timeout"
- Check your internet connection
- Verify Pinecone service status
- Try different region if persistent

---

## Managing Your Index

### View Index Stats
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("breast-guidelines")
stats = index.describe_index_stats()
print(stats)
```

### Delete Index (to save costs)
```python
pc.delete_index("breast-guidelines")
```

### Recreate Index
```bash
python build_vectorstore.py
```

---

## Alternative: Free Tier

Pinecone offers a **free tier** with limitations:
- 1 index
- 100K vectors max
- May be sufficient for testing

To use free tier:
1. Sign up for free account
2. Use same setup process
3. Monitor vector count (stay under 100K)

---

## Support

- **Pinecone Docs**: https://docs.pinecone.io/
- **Pinecone Support**: support@pinecone.io
- **Community**: https://community.pinecone.io/

---

**Ready to go!** Your vector store is now in the cloud and accessible from anywhere. 🚀
