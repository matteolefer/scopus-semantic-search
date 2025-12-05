# 🧬 Scopus AI Semantic Search Engine  
A Data Science Project for Scientific Literature Exploration

This project implements an intelligent semantic search system designed for a corpus of scientific articles (Scopus export 2018–2023). Unlike traditional keyword search, this engine uses **SBERT Embeddings** to understand the meaning of a query and **Google Gemini (LLM)** to provide intelligent summaries and relevance scoring.

The project fulfills the requirements of the **Data**, **AI**, and **Visualization** modules for the DS-ICE course.

---

## 🚀 Features

🔍 **Semantic Search**: Find relevant papers even if they don't contain the exact keywords (powered by **all-MiniLM-L6-v2**).

🤖 **AI Reranking & Insight**: Uses **Google Gemini** to analyze search results, summarize abstracts, and explain why a paper is relevant to your specific question.

📊 **Interactive Dashboard**: Visualizes publication trends, top affiliations, and active research areas over time.

🕸️ **Researcher Network**: A dynamic "Star Topology" graph to visualize an author's collaboration network (Co-authorship).

---

## 🛠️ Installation & Setup

Follow these steps to launch the application locally.

### 1. Prerequisites

Ensure you have **Python 3.9+** installed.

### 2. Clone the Repository

    git clone https://github.com/matteolefer/scopus-semantic-search.git
    cd scopus-semantic-search

### 3. Install Dependencies

Install the required Python libraries (Streamlit, PyVis, Plotly, etc.):

    pip install -r requirements.txt

### 4. Data Configuration (Critical)

⚠️ Note: The raw dataset (CSVs) is **not included** in this repository due to size constraints (>100MB).

However, the processed artifacts required to run the app are included in the `output/` folder:

- `output/articles_metadata.parquet`: cleaned dataset with metadata  
- `output/articles_embeddings.npy`: pre-computed vector embeddings for 19,000+ articles  

If these files are missing, you must regenerate them by running the Jupyter Notebook:

- Place your raw Scopus CSV in the `data/` folder  
- Run `project.ipynb` to clean data and generate embeddings  

---

## 🖥️ Usage Guide

### 1. Launch the App

Run the Streamlit server:

    streamlit run app.py

The application will open in your browser at:

    http://localhost:8501

### 2. Activate AI Features (Gemini)

To enable the AI Reranking and Summarization features:

- Obtain a free API Key from **Google AI Studio**  
- Open the Sidebar in the app (left panel)  
- Paste your key into the **"Gemini API Key"** field  
- Perform a search and click the **"✨ Analyze with AI"** button  

---

## 📂 Project Structure

| File / Folder      | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `app.py`           | Main application script. Handles the UI, Tab navigation, and visualization logic. |
| `utils.py`         | The "Engine Room". Functions for loading data (cached), computing cosine similarity, and calling the Gemini API. |
| `project.ipynb`    | The Data Pipeline. Notebook documenting EDA, cleaning, and embedding generation. |
| `output/`          | Stores processed binary files (`.npy`, `.parquet`) used by the app for fast loading. |
| `requirements.txt` | List of dependencies.                                                       |

---

## 🧠 How It Works (Step-by-Step)

### Phase 1: Data Module (Preprocessing)

- **Ingestion**: Raw CSV data from Scopus is loaded.  
- **Cleaning**: Abstracts are stripped of copyright headers (e.g., "© 2018 IEEE").  
  Rows with missing abstracts are removed to ensure quality.  
- **Vectorization**: The Sentence-BERT model (**all-MiniLM-L6-v2**) converts the Title + Abstract of every paper into a 384-dimensional dense vector.  
- **Storage**:  
  - Metadata is saved as **Parquet** (fast reading).  
  - Vectors are stored as **NumPy arrays**.  

### Phase 2: AI Module (Search & Reranking)

- **Retrieval**:  
  - When a user types a query, it is embedded into the same vector space.  
  - We calculate the **Cosine Similarity** between the query vector and all document vectors to find the top matches instantly.  

- **Augmentation (RAG)**:  
  - The top results are sent to **Google Gemini** (via API) with a custom prompt.  
  - Gemini analyzes the content and returns a structured JSON containing:  
    - a summary,  
    - and a **"Relevance Score"** out of 100.  

### Phase 3: Visualization Module

- **Analytics**:  
  - Plotly charts display temporal trends and affiliation statistics.  

- **Network Analysis**:  
  - We build a graph where nodes are authors and edges are shared papers.  
  - We use a **Star Topology** centered on a specific researcher.  
  - A physics engine (e.g., `forceAtlas2Based` or `repulsion`) prevents node overlap and helps visualize collaboration clusters clearly.  

---

## 🎓 Course Information

- **Course**: 2190513 Data Science (DS-ICE)  
- **Semester**: 2025/1  
- **Project**: Scopus Dataset Analysis & Search Engine  
