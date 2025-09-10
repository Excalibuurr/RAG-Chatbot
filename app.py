import streamlit as st
import os
from dotenv import load_dotenv
from models.embeddings import DocumentEmbedder
from models.llm import LLMAnswerer

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title="Job Search & Resume Coach", layout="centered")
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/ai.png", width=64)
    st.title("Resume Coach")
    st.markdown("---")
    st.markdown("**How to use:**\n1. Upload your resume.\n2. Upload or paste the job description.\n3. Select feedback mode.\n4. View suggestions.")
    st.markdown("---")
    st.info("Your data is processed locally and securely.")

st.title("Job Search & Resume Coach")
st.markdown("<span style='font-size:18px;'><b>Get targeted resume feedback for your next job application.</b></span>", unsafe_allow_html=True)
st.divider()

# Upload Resume and Job Description (or enter JD manually)
st.header("📄 Upload Your Documents")
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "txt"], key="resume", help="PDF or TXT format")
with col2:
    jd_option = st.radio("How would you like to provide the Job Description?", ["Upload File", "Type/Paste Description"])
    jd_file = None
    jd_text_manual = ""
    if jd_option == "Upload File":
        jd_file = st.file_uploader("Upload Job Description", type=["pdf", "txt"], key="jd", help="PDF or TXT format")
    else:
        jd_text_manual = st.text_area("Paste or type the Job Description here:", help="Paste the full job description text")
st.divider()
mode = st.radio("Select Feedback Mode", ["Concise Tips", "Detailed Rewrite"], help="Choose the type of feedback you want")

def load_text(file, is_pdf):
    path = f"temp_{file.name}"
    with open(path, "wb") as f:
        f.write(file.read())
    embedder = DocumentEmbedder()
    text = embedder.load_pdf(path) if is_pdf else open(path, encoding='utf-8').read()
    os.remove(path)
    return text, embedder

if resume_file and (jd_file or jd_text_manual.strip()):
    st.markdown("---")
    st.header("🚀 Get Your AI-Powered Feedback")
    resume_text, embedder = load_text(resume_file, resume_file.name.endswith('.pdf'))
    if jd_file:
        jd_text, _ = load_text(jd_file, jd_file.name.endswith('.pdf'))
    else:
        jd_text = jd_text_manual
    jd_sections = embedder.extract_sections(jd_text)
    with st.expander("📝 Extracted JD Sections", expanded=False):
        for sec, content in jd_sections.items():
            st.markdown(f"**{sec.title()}**\n{content}")
    with st.spinner("Fetching job market trends..."):
        trends_list = fetch_job_trends(trend_query)
    trends_text = '\n'.join(trends_list)
    answerer = LLMAnswerer()
    prompt = f"Compare the following resume and job description. Identify missing skills, suggest improvements, and incorporate current market trends.\n\nResume:\n{resume_text}\n\nJob Description:\n{jd_text}\n\nCurrent Market Trends:\n{trends_text}"
    if mode == "Concise Tips":
        prompt += "\n\nGive concise bullet-point tips."
    else:
        prompt += "\n\nRewrite the resume summary and experience sections in detail."
    with st.spinner("Generating feedback using LLM..."):
        feedback = answerer.generate_answer(prompt, [resume_text, jd_text, trends_text])
    st.success("### 🎯 AI-Powered Feedback")
    st.markdown(f"<div style='background-color:#f0f8ff;padding:16px;border-radius:8px;border:1px solid #e0e0e0;'>{feedback}</div>", unsafe_allow_html=True)
    # ...existing code...
    """Initialize and return the Groq chat model"""
    # ...existing code...


def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    
    # Get configuration from environment variables or session state
    # Default system prompt
    system_prompt = ""
    

    
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