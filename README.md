# Document QA System

A local Retrieval-Augmented Generation (RAG) system built with Python, LangChain, FAISS, and Google Gemini. This system loads documents (PDF and TXT) from a local directory, embeds them using a free local Hugging Face model (`all-MiniLM-L6-v2`), stores them in a local vector database, and allows querying them with Gemini.

---

## Setup Instructions

### 1. Create a Virtual Environment

It is highly recommended to use a virtual environment (`venv`) to keep the dependencies isolated and avoid conflicts.

#### **On Windows**
Open your terminal (PowerShell, Command Prompt, or Git Bash) and run:
```bash
python -m venv venv
```

#### **On macOS / Linux**
Open your terminal and run:
```bash
python3 -m venv venv
```

---

### 2. Activate the Virtual Environment

Before installing dependencies or running scripts, activate the virtual environment.

#### **On Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```
*Note: If you get a policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.*

#### **On Windows (CMD)**
```cmd
venv\Scripts\activate.bat
```

#### **On Windows (Git Bash)**
```bash
source venv/Scripts/activate
```

#### **On macOS / Linux**
```bash
source venv/bin/activate
```

*(You will know it is active when `(venv)` appears at the start of your terminal prompt).*

---

### 3. Install Dependencies

Once the environment is active, upgrade `pip` and install the package requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a file named `.env` in the root directory and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## How to Use

### 1. Load & Ingest Documents
Place your source documents (PDFs and `.txt` files) inside the `docs/` folder in the root directory. Then run the ingestion script to split the documents, generate embeddings, and save the database to disk:

```bash
python ingest.py
```
This generates a `faiss_index/` folder containing the serialized vector database.

### 2. Query the System
Run the interactive query script to ask questions about your documents in real time:

```bash
python query.py
```
Type your questions when prompted. To exit the interactive session, type `quit`.
