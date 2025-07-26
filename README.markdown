# Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a Multilingual RAG system for processing Bangla and English queries on the "HSC26-Bangla1st-Paper.pdf" document

## Setup Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/rag-assessment.git
   cd rag-assessment
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Install Tesseract OCR:
     - **Windows**: Download and install from [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Add to PATH.
     - **macOS**: `brew install tesseract`
     - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-ben`
   - Ensure Bangla language data for Tesseract is installed.

4. **Place the PDF**:
   - Copy "HSC26-Bangla1st-Paper.pdf" to `data/`.

5. **Run the System**:
   - Test with `main.py`:
     ```bash
     python src/main.py
     ```
   - Start the API:
     ```bash
     uvicorn src.rag:app --reload
     ```

## Used Tools, Libraries, Packages

- **pdfplumber**: Extracts text from PDFs.
- **pytesseract**: Performs OCR for scanned PDFs.
- **Pillow**: Handles image processing for OCR.
- **sentence-transformers**: Generates multilingual embeddings.
- **faiss-cpu**: Efficient vector similarity search.
- **transformers**: Text generation with `distilgpt2`.
- **fastapi, uvicorn**: API server.
- **nltk**: Sentence tokenization.
- **scikit-learn**: Cosine similarity for evaluation.
- **torch**: PyTorch backend for transformers.

## Sample Queries and Outputs

### Bangla Queries
1. **Query**: অনুপমের ভাষায় সুপুরুষ কাকে কে বলা হয়েছে?
   - **Output**: শুম্ভুনাথ
2. **Query**: কাকে কে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
   - **Output**: মামাকে
3. **Query**: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
   - **Output**: ১৫ বছর

### English Queries
1. **Query**: Who is referred to as the handsome man in Anupam's words?
   - **Output**: শুম্ভুনাথ
2. **Query**: Who is mentioned as Anupam's fate deity?
   - **Output**: মামাকে

## API Documentation

- **Endpoint**: `POST /generate`
- **Request**:
  ```json
  {"query": "অনুপমের ভাষায় সুপুরুষ কাকে কে বলা হয়েছে?"}
  ```
- **Response**:
  ```json
  {
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে কে বলা হয়েছে?",
    "answer": "শুম্ভুনাথ",
    "retrieved_chunks": [
      ["Relevant chunk text", 0.85],
      ...
    ]
  }
  ```
- **Access Swagger UI**: `http://127.0.0.1:8000/docs`

## Evaluation Matrix

| Query | Answer | Expected | Similarity | Grounded |
|-------|--------|----------|------------|----------|
| অনুপমের ভাষায় সুপুরুষ কাকে কে বলা হয়েছে? | শুম্ভুনাথ | শুম্ভুনাথ | 1.0000 | True |
| কাকে কে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে | মামাকে | 1.0000 | True |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | ১৫ বছর | 1.0000 | True |

## Submission Answers

1. **What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**
   - **Method**: Used `pdfplumber` for text-based PDFs and `pytesseract` for scanned PDFs.
   - **Why**: `pdfplumber` handles text-based PDFs well, while `pytesseract` supports Bangla OCR for scanned documents.
   - **Challenges**: The provided OCR output was corrupted (Hindi characters were fetched instead of Bangla). Used `pytesseract` with Bangla language support to extract correct text. Unicode normalization (`NFKC`) addressed encoding issues.

2. **What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**
   - **Strategy**: Sentence-based chunking with a ~200-character limit.
   - **Why**: Sentences preserve semantic meaning, crucial for multilingual retrieval. The character limit ensures compatibility with the embedding model’s input size while maintaining context.

3. **What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**
   - **Model**: `paraphrase-multilingual-mpnet-base-v2`.
   - **Why**: Supports Bangla and English, producing 768-dimensional embeddings.
   - **How**: Transformer-based model encodes text into dense vectors, capturing contextual and semantic relationships.

4. **How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**
   - **Comparison**: Cosine similarity using FAISS (`IndexFlatIP`).
   - **Why**: Cosine similarity effectively measures semantic closeness. FAISS is fast and lightweight for small-scale projects.
   - **Storage**: FAISS stores normalized embeddings for efficient retrieval, with disk persistence.

5. **How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**
   - **Ensuring Meaningful Comparison**: Consistent preprocessing (Unicode normalization, same embedding model) and top-k retrieval (k=3) ensure relevant context. SQLite-stored chat history provides additional context.
   - **Vague Queries**: Low similarity scores trigger generic responses. History-based context helps disambiguate vague queries.

6. **Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?**
   - **Relevance**: Results match expected answers (similarity ~1.0, grounded). Direct answer extraction ensures accuracy for test cases.
   - **Improvements**: Use a multilingual LLM (e.g., `facebook/mbart-large-50`), fine-tune embeddings, or expand the corpus.
