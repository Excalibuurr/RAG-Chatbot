from utils.websearch import fetch_job_trends

# Streamlit UI for Job Search and Resume Coach
import streamlit as st
import os
from models.embeddings import DocumentEmbedder
from models.llm import semantic_search, LLMAnswerer

st.set_page_config(page_title="Job Search & Resume Coach", layout="centered")
st.title("Job Search & Resume Coach Chatbot")


st.write("Upload your resume or a job description (PDF/TXT) and get tailored feedback. Choose concise tips or a detailed rewrite. Optionally, get current market trends!")

# Job market trends search
st.subheader("Job Market Trends")
default_query = "top skills for AI engineering roles in 2025"
trend_query = st.text_input("Search for current job market trends", value=default_query)
if st.button("Fetch Trends"):
    with st.spinner("Fetching latest trends..."):
        trends = fetch_job_trends(trend_query)
    st.write("**Top Results:**")
    for t in trends:
        st.write(f"- {t}")



# Upload Resume and Job Description (or enter JD manually)
st.subheader("Upload Your Documents")
resume_file = st.file_uploader("Upload Resume", type=["pdf", "txt"], key="resume")
jd_option = st.radio("How would you like to provide the Job Description?", ["Upload File", "Type/Paste Description"])
jd_file = None
jd_text_manual = ""
if jd_option == "Upload File":
    jd_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"], key="jd")
else:
    jd_text_manual = st.text_area("Paste or type the Job Description here:")
mode = st.radio("Select Feedback Mode", ["Concise Tips", "Detailed Rewrite"])


if resume_file and (jd_file or jd_text_manual.strip()):
    # Save resume temporarily
    resume_path = f"temp_{resume_file.name}"
    with open(resume_path, "wb") as f:
        f.write(resume_file.read())
    embedder = DocumentEmbedder()
    resume_text = embedder.load_pdf(resume_path) if resume_path.endswith('.pdf') else open(resume_path, encoding='utf-8').read()
    # Get JD text from file or manual input
    if jd_file:
        jd_path = f"temp_{jd_file.name}"
        with open(jd_path, "wb") as f:
            f.write(jd_file.read())
        jd_text = embedder.load_pdf(jd_path) if jd_path.endswith('.pdf') else open(jd_path, encoding='utf-8').read()
        os.remove(jd_path)
    else:
        jd_text = jd_text_manual
    resume_sections = embedder.extract_sections(resume_text)
    jd_sections = embedder.extract_sections(jd_text)
    # Only print extracted JD sections
    st.subheader("Extracted JD Sections")
    for sec, content in jd_sections.items():
        st.markdown(f"**{sec.title()}**\n{content}")
    # Fetch live job market trends
    with st.spinner("Fetching job market trends..."):
        trends_list = fetch_job_trends(trend_query)
    trends_text = '\n'.join(trends_list)
    # Use LLM to compare and suggest improvements
    answerer = LLMAnswerer()
    prompt = f"Compare the following resume and job description. Identify missing skills, suggest improvements, and incorporate current market trends.\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}\n\nCurrent Market Trends:\n{trends_text}"
    if mode == "Concise Tips":
        prompt += "\n\nGive concise bullet-point tips."
    else:
        prompt += "\n\nRewrite the resume summary and experience sections in detail."
    with st.spinner("Generating feedback using LLM..."):
        feedback = answerer.generate_answer(prompt, [resume_text, jd_text, trends_text])
    st.subheader("AI-Powered Feedback")
    st.write(feedback)
    os.remove(resume_path)

# RAG Chatbot main application
from models.embeddings import DocumentEmbedder
from models.llm import semantic_search, LLMAnswerer

def load_documents(folder: str) -> list:
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith('.pdf') or fname.endswith('.txt'):
            docs.append(os.path.join(folder, fname))
    return docs

def main():
    print("Initializing RAG Chatbot...")
    embedder = DocumentEmbedder()
    answerer = LLMAnswerer()
    doc_folder = '.'
    doc_files = load_documents(doc_folder)
    embedded_docs = []
    for f in doc_files:
        print(f"Processing {f}...")
        embedded_docs.extend(embedder.process_document(f))
    print("Ready to chat! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        top_chunks = semantic_search(query, embedded_docs, embedder)
        answer = answerer.generate_answer(query, [c['chunk'] for c in top_chunks])
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    main()

def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the chat model"""
    try:
        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.1-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    
    # Get configuration from environment variables or session state
    # Default system prompt
    system_prompt = ""
    
    
    # Determine which provider to use based on available API keys
    chat_model = get_chatgroq_model()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    # if chat_model:
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("🔧 No API keys found in environment variables. Please check the Instructions page to set up your API keys.")

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()