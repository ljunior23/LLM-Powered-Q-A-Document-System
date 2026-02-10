#  RAG Q&A System with OpenAI & FAISS

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, OpenAI API, FAISS vector store, and Streamlit. Upload your PDF documents and ask questions with AI-powered answers backed by your documents.

![RAG System](https://img.shields.io/badge/AI-RAG%20System-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)

##  Features

-  **PDF Document Processing** - Upload and process PDF documents automatically
-  **Intelligent Search** - FAISS vector store for fast semantic search
-  **AI-Powered Answers** - GPT-3.5/GPT-4 powered responses
-  **Quality Metrics** - RAGAS evaluation for faithfulness and relevancy
-  **Chat Interface** - Interactive chat with conversation history
-  **Cost Tracking** - Monitor token usage and API costs in real-time
-  **Export Functionality** - Download chat history and metrics
-  **Configurable** - Adjustable chunk size, overlap, and model parameters

##  Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
```bash
   git clone (https://github.com/ljunior23/LLM-Powered-Q-A-Document-System)
   cd rag-openai-faiss
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the application**
```bash
   streamlit run rag_openai_faiss.py
```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 📖 Usage

1. **Enter your OpenAI API Key** in the sidebar
2. **Upload a PDF document** using the file uploader
3. **Wait for processing** - The system will chunk and embed your document
4. **Ask questions** in the chat interface
5. **View sources** - Expand to see which document sections were used
6. **Check metrics** - View faithfulness and relevancy scores for each answer

##  Configuration Options

### Model Settings
- **OpenAI Model**: Choose between GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **Temperature**: Control response creativity (0.0 - 1.0)

### RAG Settings
- **Chunk Size**: Size of text chunks (500-2000 characters)
- **Chunk Overlap**: Overlap between chunks (0-500 characters)
- **Retrieved Chunks (k)**: Number of relevant chunks to retrieve (1-10)

### Evaluation
- **Enable/Disable Metrics**: Toggle RAGAS evaluation
- **Metrics Tracked**: Faithfulness, Answer Relevancy

##  Evaluation Metrics

The system uses [RAGAS](https://github.com/explodinggradients/ragas) for answer quality evaluation:

- **Faithfulness** (0-1): Measures if the answer is grounded in the retrieved context
- **Answer Relevancy** (0-1): Measures if the answer is relevant to the question

##  Cost Estimation

Approximate costs per 1K tokens (as of 2024):

| Model | Input | Output | Average |
|-------|-------|--------|---------|
| GPT-4 | $0.03 | $0.06 | $0.045 |
| GPT-4-turbo | $0.01 | $0.03 | $0.02 |
| GPT-3.5-turbo | $0.0005 | $0.0015 | $0.001 |

The app tracks your usage in real-time in the sidebar.

##  Architecture
```
┌─────────────┐
│   PDF Doc   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Text Split │ (RecursiveCharacterTextSplitter)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embeddings │ (OpenAI text-embedding-ada-002)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ FAISS Store │ (Vector Database)
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│   Question  │────▶│  Retrieval  │
└─────────────┘     └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  GPT Model  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Answer    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    RAGAS    │ (Evaluation)
                    └─────────────┘
```

##  Technology Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **LLM Provider**: OpenAI (GPT-3.5/GPT-4)
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF
- **Evaluation**: RAGAS
- **Language**: Python 3.9+

##  Project Structure
```
rag-openai-faiss/
├── rag_openai_faiss.py    # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore file
```

##  Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file: `rag_openai_faiss.py`
   - Click "Deploy"

### Deploy to Other Platforms

**Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "rag_openai_faiss.py"]
```

**Heroku**
```bash
heroku create your-app-name
git push heroku main
```

##  Security Notes

-  **Never commit your API keys** to version control
- Use environment variables or Streamlit secrets for API keys
- The `.gitignore` file is configured to exclude sensitive files
- API keys are only stored in memory during the session

##  Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install --upgrade langchain langchain-openai langchain-community
```

**2. FAISS Installation Issues**
```bash
# On Windows, if faiss-cpu fails:
pip install faiss-cpu==1.7.4
```

**3. OpenAI API Errors**
- Check your API key is valid
- Ensure you have credits in your OpenAI account
- Verify API key starts with `sk-`

**4. PDF Processing Errors**
```bash
pip install pypdf --upgrade
```

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Your Name - kwaleon@umich.edu

Project Link: [https://github.com/ljunior23/LLM-Powered-Q-A-Document-System](https://github.com/ljunior23/LLM-Powered-Q-A-Document-System)

##  Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [OpenAI](https://openai.com/) - GPT models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Streamlit](https://streamlit.io/) - Web framework
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG evaluation framework

##  Metrics & Performance

- **Average Response Time**: 3-5 seconds
- **Embedding Time**: ~1-2 seconds per page
- **Supported Document Size**: Up to 100 pages recommended
- **Concurrent Users**: Depends on deployment platform

##  Future Enhancements

- [ ] Multi-document support
- [ ] Document management (delete/update)
- [ ] Conversation memory across sessions
- [ ] Support for more file formats (DOCX, TXT, HTML)
- [ ] Advanced filtering and search
- [ ] User authentication
- [ ] Usage analytics dashboard
- [ ] Batch processing
- [ ] API endpoint

---

**Built with ❤️ using LangChain, OpenAI, FAISS, and Streamlit**
