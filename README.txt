# 🚀 Semantic Cache RAG: Efficient Question-Answering System

---

## 1. Introduction

This project is a practical implementation of the **Semantic Caching** system proposed in the scientific paper: "Semantic Caching of Contextual Summaries for Efficient Question-Answering with Language Models."

The main goal is to build an intelligent **Question-Answering (QA)** system capable of **minimizing latency and computational costs** by preventing redundant calls to the Large Language Model (LLM).

The system operates on a **"cache-first"** principle: it checks a **semantic cache** built on **Redis** before executing the full **RAG (Retrieval-Augmented Generation)** pipeline.

**[Original Paper Link](https://arxiv.org/pdf/2505.11271)**

---

## 2. Key Features

### Cache-First Logic
* **Semantic Cache:** Stores question-context-answer pairs in a **Redis Hash** for fast reuse.
* **Vector Search:** Utilizes the **RediSearch** module to create vector indexes, allowing search for **semantically similar** questions rather than exact keyword matches.

### Core QA Capabilities
* **RAG Mode:** Enables in-depth question-answering based on the content of an uploaded **PDF document**.
* **General Chatbot Mode:** Allows conventional conversation leveraging the LLM's general knowledge base while simultaneously populating and utilizing the semantic cache.

### Technology Stack
* **Embedding Support:** Uses the high-performance **all-MiniLM-L6-v2** model (384 dimensions) to generate semantic vectors for questions.
* **Docker Deployment:** The entire system, including Redis, is easily launched with a single `docker-compose up` command.

---

## 3. Technology Stack

The system is built using modern and robust components:

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM Brain** | **Google Gemini** | The core model for generation and reasoning. |
| **Semantic Cache** | **Redis** (with RediSearch) | High-speed memory store and vector index for semantic similarity search. |
| **Integration Framework** | **LangChain** | Orchestrates the RAG, Caching, and LLM connection pipelines. |
| **Embedding Model** | **all-MiniLM-L6-v2** | Generates 384-dimensional vector representations. |
| **User Interface** | **Streamlit** | Provides the interactive web interface for file upload and chat. |
| **Deployment** | **Docker** | Ensures easy, portable, and reproducible setup. |

---

## 4. Getting Started: Setup and Installation

### System Requirements
* **Docker** and **Docker Compose**
* **Python 3.10+**
* A **Google Gemini API Key**

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [URL_CUA_REPO_CUA_BAN]
    cd [TEN_REPO_CUA_BAN]
    ```

2.  **Create a virtual environment and install libraries:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Use .\venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    * Create a file named **`.env`** in the root directory.
    * Add your API key inside the file as follows:
    ```
    MY_GEMINI_API_KEY="..................."
    ```

4.  **Launch the system with Docker:**
    * Ensure **Docker Desktop** is running.
    * Run the following command to start both the **Redis** container and the **Streamlit** application:
    ```bash
    docker-compose up --build
    ```

---

## 5. Usage Guide

Once the system is running via `docker-compose`, access the Streamlit interface in your web browser (usually at `http://localhost:8501`).

### Chatbot Mode (Default)
* You can immediately start chatting. All questions and corresponding answers will be automatically stored in the semantic cache.

### RAG Mode (Document-Specific QA)
1.  **Upload Document:** Use the left sidebar to upload a **PDF file**.
2.  **Process:** Click the **"Xử lý tài liệu"** (Process Document) button.
3.  **Query:** Once processing is complete, you can begin asking questions specifically related to the content of the uploaded PDF file.
