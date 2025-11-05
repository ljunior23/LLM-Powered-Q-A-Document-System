import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import time
from datetime import datetime

# Evaluation metrics
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System with OpenAI",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG Q&A System (OpenAI + FAISS)")
st.markdown("""
This system uses **OpenAI API** with **FAISS** vector store for Retrieval-Augmented Generation.
Includes **evaluation metrics** for faithfulness and relevance assessment.
""")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key input
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key from https://platform.openai.com/api-keys"
    )
    
    # Validate and set API key
    api_key_valid = False
    if openai_api_key:
        if openai_api_key.startswith('sk-'):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            api_key_valid = True
            st.success("✅ API Key configured")
        else:
            st.error("❌ Invalid API key format. Should start with 'sk-'")
    else:
        st.warning("⚠️ Enter your OpenAI API key above")
    
    st.markdown("---")
    
    # Model selection
    st.header("🤖 Model Settings")
    model_name = st.selectbox(
        "OpenAI Model",
        ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        index=2,
        help="GPT-4 is more accurate but expensive. GPT-3.5-turbo is fast and cost-effective."
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.markdown("---")
    
    # File upload
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to create your knowledge base"
    )
    
    # Processing parameters
    st.header("🔧 RAG Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
    k_results = st.slider("Retrieved Chunks (k)", 1, 10, 4, 1)
    
    st.markdown("---")
    
    # Evaluation settings
    st.header("📊 Evaluation")
    enable_evaluation = st.checkbox("Enable Metrics Evaluation", value=RAGAS_AVAILABLE)
    
    if enable_evaluation and RAGAS_AVAILABLE:
        st.info("Evaluates: Faithfulness, Answer Relevancy")
    elif enable_evaluation and not RAGAS_AVAILABLE:
        st.warning("Install ragas to enable evaluation: pip install ragas")
    
    st.markdown("---")
    
    # Usage stats
    st.header("💰 Usage Stats")
    st.metric("Total Tokens", st.session_state.total_tokens)
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.metrics_history = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# Functions
def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    """Process the uploaded PDF and create vector store"""
    
    # Ensure API key is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OpenAI API key not set. Please enter it in the sidebar.")
    
    with st.spinner("📚 Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings using OpenAI
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
            
            # Create vector store
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            return vector_store, len(chunks)
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

def create_qa_chain(vector_store, k_results, model_name, temperature):
    """Create the QA chain with retrieval using LCEL (LangChain Expression Language)"""
    
    # Create ChatOpenAI LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": k_results})
    
    # Custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
Be specific and detailed in your answer, citing relevant information from the context.

Context: {context}

Question: {question}

Detailed Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the RAG chain using LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Wrapper class to provide consistent interface
    class RAGChainWrapper:
        def __init__(self, chain, retriever):
            self.chain = chain
            self.retriever = retriever
        
        def invoke(self, inputs):
            """Handle both 'input' and 'query' keys"""
            question = inputs.get("input") or inputs.get("query", "")
            answer = self.chain.invoke(question)
            # FIXED: Use invoke() instead of get_relevant_documents()
            docs = self.retriever.invoke(question)
            return {
                "answer": answer,
                "result": answer,  # For backwards compatibility
                "context": docs,
                "source_documents": docs  # For backwards compatibility
            }
        
        def __call__(self, inputs):
            """Allow calling as a function"""
            return self.invoke(inputs)
    
    return RAGChainWrapper(rag_chain, retriever)

def evaluate_response(question, answer, contexts, ground_truth=None):
    """Evaluate response using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {}
    
    try:
        # Ensure OpenAI API key is available for RAGAS
        if "OPENAI_API_KEY" not in os.environ:
            return {}
        
        # Prepare data for evaluation
        # Make sure contexts are strings
        context_list = []
        for ctx in contexts:
            if hasattr(ctx, 'page_content'):
                context_list.append(ctx.page_content)
            else:
                context_list.append(str(ctx))
        
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [context_list],
        }
        
        # Create dataset
        dataset = Dataset.from_dict(data)
        
        # Set up OpenAI for RAGAS evaluation
        from langchain_openai import ChatOpenAI
        eval_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Only use metrics that don't require ground truth
        metrics_to_use = [faithfulness, answer_relevancy]
        
        # Evaluate with the LLM
        result = evaluate(
            dataset,
            metrics=metrics_to_use,
            llm=eval_llm,
            embeddings=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
        )
        
        # Convert EvaluationResult to dictionary
        metrics_dict = {}
        try:
            if hasattr(result, 'to_pandas'):
                # Convert to pandas and then to dict
                df = result.to_pandas()
                if not df.empty:
                    # Get the first row as dict
                    row_dict = df.iloc[0].to_dict()
                    # Filter out NaN values
                    metrics_dict = {k: v for k, v in row_dict.items() if v == v}  # v == v is False for NaN
            elif isinstance(result, dict):
                metrics_dict = {k: v for k, v in result.items() if v == v}
            else:
                # Try to access as attributes
                faith = getattr(result, 'faithfulness', None)
                rel = getattr(result, 'answer_relevancy', None)
                if faith is not None and faith == faith:  # Check not NaN
                    metrics_dict['faithfulness'] = faith
                if rel is not None and rel == rel:  # Check not NaN
                    metrics_dict['answer_relevancy'] = rel
        except Exception as e:
            st.warning(f"Error converting metrics: {e}")
        
        return metrics_dict
    except Exception as e:
        # Show error in development, hide in production
        import traceback
        error_msg = f"Evaluation error: {str(e)}\n{traceback.format_exc()}"
        st.warning(f"Evaluation failed: {str(e)}")
        return {}

def calculate_cost(tokens_used, model_name):
    """Calculate approximate cost based on tokens and model"""
    # Pricing as of 2024 (per 1K tokens)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    }
    
    if model_name in pricing:
        # Approximate 50/50 split between input and output
        avg_price = (pricing[model_name]["input"] + pricing[model_name]["output"]) / 2
        return (tokens_used / 1000) * avg_price
    return 0.0

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat")
    
    # Check for API key
    if not api_key_valid:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.info("""
        **How to get an API key:**
        1. Go to https://platform.openai.com/api-keys
        2. Sign up or log in
        3. Create a new API key
        4. Copy and paste it in the sidebar
        """)
    else:
        # Process PDF if uploaded
        if uploaded_file and st.session_state.vector_store is None:
            try:
                vector_store, num_chunks = process_pdf(uploaded_file, chunk_size, chunk_overlap)
                st.session_state.vector_store = vector_store
                st.session_state.qa_chain = create_qa_chain(
                    vector_store, k_results, model_name, temperature
                )
                st.success(f"✅ Document processed! Created {num_chunks} chunks.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.exception(e)
        
        # Display chat history
        for i, (question, answer, metrics) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
                
                # Show metrics if available
                if metrics and enable_evaluation and isinstance(metrics, dict):
                    with st.expander("📊 Response Metrics"):
                        cols = st.columns(2)
                        
                        # Safely access metrics
                        if 'faithfulness' in metrics and metrics['faithfulness'] is not None:
                            try:
                                cols[0].metric("Faithfulness", f"{float(metrics['faithfulness']):.3f}")
                            except (ValueError, TypeError):
                                pass
                        
                        if 'answer_relevancy' in metrics and metrics['answer_relevancy'] is not None:
                            try:
                                cols[1].metric("Relevancy", f"{float(metrics['answer_relevancy']):.3f}")
                            except (ValueError, TypeError):
                                pass
        
        # Chat input
        if st.session_state.vector_store:
            question = st.chat_input("Ask a question about your document...")
            
            if question:
                # Add user message to chat
                with st.chat_message("user"):
                    st.write(question)
                
                # Get answer
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            start_time = time.time()
                            
                            # Get response with token tracking
                            with get_openai_callback() as cb:
                                response = st.session_state.qa_chain.invoke({"input": question})
                                
                                # Update usage stats
                                st.session_state.total_tokens += cb.total_tokens
                                cost = calculate_cost(cb.total_tokens, model_name)
                                st.session_state.total_cost += cost
                            
                            answer = response.get('answer', response.get('result', ''))
                            source_docs = response.get('context', response.get('source_documents', []))
                            
                            response_time = time.time() - start_time
                            
                            st.write(answer)
                            
                            # Evaluate response
                            metrics = {}
                            if enable_evaluation and RAGAS_AVAILABLE and source_docs:
                                with st.spinner("Evaluating response quality..."):
                                    eval_result = evaluate_response(
                                        question, 
                                        answer, 
                                        source_docs
                                    )
                                    if eval_result:
                                        metrics = eval_result
                                        
                                        # Display metrics
                                        with st.expander("📊 Response Metrics"):
                                            cols = st.columns(3)
                                            
                                            # Safely access metrics
                                            if 'faithfulness' in metrics and metrics['faithfulness'] is not None:
                                                try:
                                                    cols[0].metric("Faithfulness", f"{float(metrics['faithfulness']):.3f}")
                                                except (ValueError, TypeError):
                                                    pass
                                            
                                            if 'answer_relevancy' in metrics and metrics['answer_relevancy'] is not None:
                                                try:
                                                    cols[1].metric("Relevancy", f"{float(metrics['answer_relevancy']):.3f}")
                                                except (ValueError, TypeError):
                                                    pass
                                            
                                            cols[2].metric("Time", f"{response_time:.2f}s")
                            
                            # Show sources
                            if source_docs:
                                with st.expander("📚 View Sources"):
                                    for i, doc in enumerate(source_docs[:3]):
                                        st.markdown(f"**Source {i+1}:**")
                                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                        st.text(content[:300] + "...")
                                        st.markdown("---")
                            
                            # Add to chat history
                            st.session_state.chat_history.append((question, answer, metrics))
                            st.session_state.metrics_history.append({
                                'timestamp': datetime.now(),
                                'question': question,
                                'response_time': response_time,
                                'tokens': cb.total_tokens,
                                'cost': cost,
                                **metrics
                            })
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
                            st.exception(e)
                            if "API key" in str(e) or "authentication" in str(e).lower():
                                st.error("Please check your OpenAI API key.")
        else:
            st.info("👈 Please upload a PDF document in the sidebar to start chatting!")

with col2:
    st.header("📊 System Info")
    
    if st.session_state.vector_store:
        st.metric("Status", "✅ Ready")
        st.metric("Chat Messages", len(st.session_state.chat_history))
        st.metric("Retrieved Chunks", k_results)
        st.metric("Model", model_name)
        
        st.markdown("---")
        st.markdown("**Document Details:**")
        st.write(f"- Chunk Size: {chunk_size}")
        st.write(f"- Chunk Overlap: {chunk_overlap}")
        st.write(f"- Embedding: text-embedding-ada-002")
        st.write(f"- Vector DB: FAISS")
        
        # Average metrics
        if st.session_state.metrics_history and enable_evaluation:
            st.markdown("---")
            st.markdown("**Average Metrics:**")
            
            avg_metrics = {}
            for metric_name in ['faithfulness', 'answer_relevancy']:
                values = []
                for m in st.session_state.metrics_history:
                    if isinstance(m, dict) and metric_name in m:
                        try:
                            val = float(m[metric_name])
                            if val is not None and not (val != val):  # Check for None and NaN
                                values.append(val)
                        except (ValueError, TypeError):
                            pass
                
                if values:
                    avg_metrics[metric_name] = sum(values) / len(values)
            
            if avg_metrics:
                for name, value in avg_metrics.items():
                    st.metric(name.replace('_', ' ').title(), f"{value:.3f}")
        
    else:
        st.metric("Status", "⏳ Waiting for document")
        st.info("Upload a PDF to see system information")
    
    st.markdown("---")
    st.markdown("""
    **RAG Pipeline:**
    1. 📄 Document upload
    2. 🔪 Text chunking
    3. 🧮 OpenAI embeddings
    4. 💾 FAISS storage
    5. ❓ Question embedding
    6. 🔍 Similarity search
    7. 🤖 GPT answer generation
    8. 📊 Quality evaluation
    """)

# Footer with export functionality
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.metrics_history:
        if st.button("📥 Export Metrics"):
            import json
            metrics_json = json.dumps(
                [{**m, 'timestamp': m['timestamp'].isoformat()} 
                 for m in st.session_state.metrics_history],
                indent=2
            )
            st.download_button(
                "Download Metrics JSON",
                metrics_json,
                "metrics.json",
                "application/json"
            )

with col2:
    if st.session_state.chat_history:
        if st.button("💾 Export Chat"):
            chat_text = "\n\n".join([
                f"Q: {q}\nA: {a}\n{'-'*50}"
                for q, a, _ in st.session_state.chat_history
            ])
            st.download_button(
                "Download Chat History",
                chat_text,
                "chat_history.txt",
                "text/plain"
            )

with col3:
    st.markdown(f"**Vector Store:** FAISS | **LLM:** OpenAI {model_name}")

st.markdown("""
<div style='text-align: center'>
    <p>Built with LangChain, OpenAI API, FAISS, and Streamlit</p>
    <p>Evaluation powered by RAGAS framework</p>
</div>
""", unsafe_allow_html=True)