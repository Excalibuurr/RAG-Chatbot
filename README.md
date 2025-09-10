# Job Search & Resume Coach Chatbot

A Retrieval-Augmented Generation (RAG) powered chatbot to help job seekers tailor their resumes and prepare for interviews using AI and live job market trends.

## Features
- Upload your resume (PDF/TXT)
- Upload or paste a job description (PDF/TXT or text)
- Fetch current job market trends via web search
- Compare resume and job description using semantic search and LLM
- Get concise tips or detailed rewrite suggestions
- Secure API key management via `.env`

## How It Works
1. **Document Ingestion:** Your resume and job description are processed and chunked for semantic analysis.
2. **Embeddings:** Text chunks are embedded using `all-MiniLM-L6-v2` for similarity search.
3. **Web Search:** Fetches live job market trends to inform feedback.
4. **LLM Feedback:** Uses `meta-llama/Llama-3-8B-Instruct` to generate actionable feedback and rewrites.

## Setup
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root:
   ```env
   API_KEY=your_huggingface_api_key
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Enter a job market trend query and fetch trends.
- Upload your resume and job description (or paste JD text).
- Select feedback mode (Concise Tips or Detailed Rewrite).
- View AI-powered feedback and suggestions.

## File Structure
- `app.py` — Streamlit UI and main logic
- `models/embeddings.py` — Document ingestion and embedding
- `models/llm.py` — LLM integration
- `utils/websearch.py` — Web search utility
- `requirements.txt` — Python dependencies
- `.env.example` — API key template

## Example Files
- `example_resume.txt` — Sample resume for an AI Engineer
- `example_jd.txt` — Sample job description for an AI Engineer role

Use these files to test the app by uploading them in the Streamlit interface.

## Example API Key Setup
```
API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Notes
- Make sure your Hugging Face account has access to the selected LLM model.
- For best results, use clear and relevant resume and job description files.

## License
MIT
