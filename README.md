[readme.txt](https://github.com/user-attachments/files/17007323/readme.txt)
## 1.Project Overview -----------------------------------------------------
This project creates a "business policy chatbot" that answers user queries or processes PDFs to retrieve relevant information. 
The chatbot uses "Cohere" for text embeddings (transforming text into vectors) and "Pinecone" for storing and searching those embeddings based on 
user queries. Finally, we use "Streamlit" to build an interactive user interface (UI) for users to ask questions or upload PDFs.


## 2.Setting up the environment ----

### Install Required Libraries-----
You need to install the required libraries in your Jupyter environment:

pip install cohere pinecone-client PyPDF2 streamlit


These libraries include:
- Cohere - For generating text embeddings.
- Pinecone - For storing and querying vectorized text.
- PyPDF2 - For reading PDF files.
- Streamlit - For creating a web app.

---

## 3.Jupyter Notebook (Code for Embedding and Indexing Data)========

### Step 1 - Import Required Libraries-
In the notebook, import the necessary libraries and set up your API keys-

import cohere
import pinecone
import PyPDF2

# Initialize Cohere API and Pinecone
cohere_api_key = "Your-Cohere-API-Key"
pinecone_api_key = "Your-Pinecone-API-Key"
pinecone_env = "us-east-1"
co = cohere.Client(cohere_api_key)
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


### Step 2 - Create or Connect to Pinecone Index -
Pinecone is used to store our vectorized data for querying. Here's how to create and connect to a Pinecone index-

index_name = 'qa-chatbot'
dimension = 4096
index = pinecone.Index(index_name)

# Check if the index exists and create it if necessary
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")


### Step 3 - Prepare and Embed Documents - 
Now, you add some general business documents and convert them into embeddings using Cohere:

documents = [
    "Our HR policies include flexible working hours for all employees.",
    "The IT department is responsible for hardware and software maintenance.",.......................................]

# Chunk text into manageable sizes and embed using Cohere
response = co.embed(texts=documents, model="embed-english-v2.0")
embeddings = response.embeddings

# Prepare vectors for Pinecone
vectors = [(f"doc_{i}", embeddings[i]) for i in range(len(documents))]

# Upsert into Pinecone index
index.upsert(vectors=vectors)

### Step 4: Query Pinecone with User Input --
When a user asks a question, convert it to a vector and retrieve similar documents:

def search_query(query)---------
    query_embedding = co.embed(texts=[query], model="embed-english-v2.0").embeddings[0]
    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    return result




## 4. Building the Streamlit App (app.py)----------------------------

### Step 1: Set up Streamlit and Input Fields
Now, we build the UI using Streamlit:

import streamlit as st

# Streamlit title
st.title("Business Policy & Operations Chatbot")

# Text input for user questions
user_question = st.text_input("Enter your question:")


### Step 2: Handle PDF Uploads---------------
Allow users to upload PDF files----

uploaded_file = st.file_uploader("Or upload a PDF file:", type="pdf")

# Process the uploaded PDF if available
if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    pdf_text = "".join([page.extract_text() for page in reader.pages])
    st.write("Extracted text:", pdf_text[:2000])  # Display first 2000 characters


### Step 3: Query Pinecone and Generate Responses------------
Use the search function to return relevant documents and generate a response:

if st.button("Submit"):
    if user_question:
        retrieved_texts = search_query(user_question)
        context = " ".join([match['metadata']['text'] for match in retrieved_texts])
        st.write("Chatbot Response:", context)




## 5.Deploying the App------

### Step 1: Create an app.py file---
Save all the code into an `app.py` file.

### Step 2: Deploy on Streamlit Cloud-----
1. Push your code to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub repository.
3. Click "Deploy" to launch the chatbot.


## 6.Overview of the Project-----------------------------------------------------------

- Goal --------  Build a chatbot that answers business-related questions or processes PDFs using text embeddings and vector search.
- Libraries -- 
   - Cohere - for text embeddings.
   - Pinecone - for storing and querying vectors.
   - PyPDF2 - for extracting text from PDFs.
   - Streamlit - for building an easy-to-use web app.
   
- Process - 
   - Prepare documents and embed them.
   - Use Pinecone to store embeddings.
   - Allow users to query the data via questions or PDF uploads.
   - Display relevant information as responses.



## 7.Challenges Faced -----------------

### 1. Understanding APIs -------
   - As a beginner, it might be challenging to understand how different APIs like Cohere and Pinecone work. 
     Reading documentation and examples helped in overcoming this.

### 2. Handling Errors ----
   - While working with Pinecone and Streamlit, handling and debugging errors was difficult at first. 
     Gradually, improving error-handling techniques helped smooth the development process.

### 3. Deploying the Application----
   - For a fresher, learning how to deploy a Python application on Streamlit Cloud was a new experience. 
     However, the user-friendly nature of Streamlit made it easier after some practice.

### 4. Text Embedding Concepts----
   - Concepts like vector embeddings and cosine similarity were unfamiliar at the start, 
     but understanding these core AI principles was crucial for the project’s success.



This project helps build a solid understanding of how modern machine learning tools can be combined to create a chatbot 
that handles both natural language queries and PDF content, helping you develop strong foundational skills in AI and web app development.
