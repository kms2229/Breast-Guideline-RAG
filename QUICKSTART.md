# Quick Start Guide - Breast Guideline RAG System (Pinecone Cloud)

## Complete Setup in 6 Steps

### Step 1: Navigate to Project Directory

```bash
cd /Users/kavachshah/Downloads/degrees/PROJ__NLP/RA/Breast-Guideline-RAG
```

### Step 2: Run Setup Script

```bash
./setup.sh
```

This will:
- Create virtual environment
- Install all dependencies

### Step 3: Get Pinecone Account

1. Go to https://www.pinecone.io/
2. Sign up (free trial or Starter plan $70/month)
3. Get your API key from dashboard
4. Copy it (starts with `pcsk_...`)

### Step 4: Add Your API Keys

Copy the example file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```bash
nano .streamlit/secrets.toml
```

Add both API keys:

```toml
OPENAI_API_KEY = "sk-your-openai-key-here"
PINECONE_API_KEY = "pcsk-your-pinecone-key-here"
PINECONE_INDEX_NAME = "breast-guidelines"
```

Save and close.

### Step 5: Build & Upload to Pinecone

**IMPORTANT**: This processes all 500+ PDFs and uploads to Pinecone (15-30 minutes).

```bash
# Make sure venv is activated
source venv/bin/activate

# Build and upload to Pinecone
python build_vectorstore.py
```

You'll see progress like:
```
Loading documents from: /Users/kavachshah/.../Breast Guideline PDFs
Loaded: 00000637-200905000-00024.pdf
...
Created 15000 text chunks
Creating Pinecone index: breast-guidelines
Uploading to Pinecone...
✅ Upload complete!
```

**Cost**: ~$2-5 for embeddings

### Step 6: Run the Application

```bash
streamlit run Chat_UI.py
```

The app opens at `http://localhost:8501`

## First Time Usage

1. **Enter API Key** in sidebar (will be validated)
2. **Select Model** (recommend `gpt-4o-mini` for cost-effectiveness)
3. **Set Top-K** to 3 (default)
4. **Ask a Question**:
   - "What are the guidelines for nipple-sparing mastectomy?"
   - "Compare DIEP flap vs implant reconstruction"
   - "What are complications of prepectoral reconstruction?"

## Troubleshooting

### "Command not found: ./setup.sh"
```bash
chmod +x setup.sh
./setup.sh
```

### "No module named 'streamlit'"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Index not found" or "PINECONE_API_KEY not found"
- Check `.streamlit/secrets.toml` has Pinecone API key
- Run `python build_vectorstore.py` to create index
- Verify index exists at https://app.pinecone.io/

### "API rate limit exceeded"
Wait 1 minute, then try again. Or use GPT-3.5-turbo.

## System Requirements

- **Python**: 3.11 or 3.12
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 200MB (just for JSON file, vectors in cloud)
- **Internet**: Required (for Pinecone & API calls)
- **Pinecone Account**: Starter plan ($70/month) or free tier

## Cost Estimates

### One-Time Setup
- **Embeddings**: ~$2-5 (500 PDFs)
- **Time**: 15-30 minutes

### Monthly Costs
- **Pinecone**: $70/month (Starter plan)
- **OpenAI** (100 queries): $1-6
- **Total**: ~$75/month

### Per Query
- **GPT-4**: ~$0.06
- **GPT-4o-mini**: ~$0.01
- **GPT-3.5-turbo**: ~$0.005

## What's Included

✅ 500+ breast cancer guideline PDFs processed  
✅ Semantic search with Pinecone cloud  
✅ Query decomposition (5 sub-questions)  
✅ Conversation history  
✅ Multiple LLM support  
✅ Streaming responses  
✅ Expandable UI sections  
✅ Accessible from anywhere  

## Next Steps

- Read `README.md` for detailed documentation
- Explore different LLM models
- Adjust Top-K for retrieval quality
- Try complex multi-part questions

## Support

If you encounter issues:
1. Check `.streamlit/secrets.toml` has valid API keys (OpenAI + Pinecone)
2. Verify Pinecone index exists at https://app.pinecone.io/
3. Ensure virtual environment is activated
4. Check `Dataset/breast_guidelines_full_text.json` exists
5. See `PINECONE_SETUP.md` for detailed Pinecone help

---

**Ready to use!** 🚀

Ask questions about breast cancer treatment guidelines and get comprehensive, evidence-based answers from 500+ research papers.
