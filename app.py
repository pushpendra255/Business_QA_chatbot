import streamlit as st
import cohere
import pinecone
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
import os

# Set up API keys
cohere_api_key = "Ur2O1JW9qQfbjO4c2yO0wGdwSdVdAXAQVn7FJ2Fk"
pinecone_api_key = "b7424a2d-a7fc-4be3-b1a6-fadeccc1953d"
pinecone_env = "us-east-1"

# Initialize Cohere
co = cohere.Client(cohere_api_key)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Define index parameters
index_name = 'qa-chatbot'
dimension = 4096
metric = 'cosine'

# Check if the index already exists and create it if necessary
try:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_env
            )
        )
except Exception as e:
    st.error(f"Error checking or creating index: {e}")

# Connect to the index
index = pc.Index(index_name)

# Prepare data for indexing
general_documents = [
    "Our HR policies include flexible working hours for all employees.",
    "The IT department is responsible for hardware and software maintenance.",
    "We offer health insurance to all full-time employees.",
    "The marketing team handles social media, advertising, and customer engagement.",
    "The company offers a 30-day return policy for all electronic items.",
    "Our finance department manages budgeting, accounting, and payroll.",
    "The customer service team provides support through phone, email, and chat.",
    "We have a dedicated legal team to handle contracts and compliance issues.",
    "Our procurement department sources materials and negotiates supplier contracts.",
    "The R&D team focuses on product innovation and development.",
    "We conduct annual performance reviews to assess employee progress and development.",
    "The company offers professional development opportunities and training programs.",
    "We have a robust data security policy to protect company and customer information.",
    "Our sales team is responsible for driving revenue through client acquisition and retention.",
    "The logistics department manages inventory, warehousing, and distribution.",
    "We have a corporate social responsibility program that supports community initiatives.",
    "The business development team identifies new market opportunities and partnerships.",
    "Our operations team ensures the efficiency of day-to-day business processes.",
    "We offer employee wellness programs, including fitness memberships and mental health resources.",
    "The administrative staff handles office management, scheduling, and administrative support.",
    "We follow a strict confidentiality policy for handling sensitive company information.",
    "The project management office oversees project planning, execution, and delivery.",
    "We have a customer feedback system to continuously improve our products and services.",
    "The quality assurance team ensures that all products meet our quality standards.",
    "Our supply chain management team coordinates the flow of goods from suppliers to customers.",
    "We have a business continuity plan to ensure operations can continue during disruptions.",
    "The risk management team identifies and mitigates potential business risks.",
    "Our IT security team implements measures to protect against cyber threats and data breaches.",
    "We offer a range of employee benefits, including retirement plans and bonuses.",
    "The legal department ensures compliance with industry regulations and laws.",
    "We conduct market research to understand customer needs and industry trends.",
    "The product management team oversees product development and lifecycle management.",
    "We provide training programs for employees to enhance their skills and knowledge.",
    "The business strategy team develops and implements long-term business goals and plans.",
    "Our customer support team handles inquiries, complaints, and service requests.",
    "We maintain partnerships with key stakeholders and industry organizations.",
    "The IT infrastructure team manages the company's network, servers, and hardware.",
    "We have a protocol for managing and responding to customer complaints and issues.",
    "Our data analytics team provides insights to support decision-making and strategy.",
    "We have a code of conduct outlining ethical standards and behavior expectations."
]

queries_and_responses = [
    ("Hello", "Hi there! How can I help you today?"),
    ("How can I start using the service?", "To start using the service, follow these steps: ..."),
]

documents = general_documents + [q for q, _ in queries_and_responses] + [r for _, r in queries_and_responses]

chunk_size = 1536

def chunk_text(text, chunk_size=1536):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunks = []
for document in documents:
    chunks.extend(chunk_text(document, chunk_size))

# Embed chunks
try:
    response = co.embed(texts=chunks, model="embed-english-v2.0")
    metadata = [{"text": chunk} for chunk in chunks]
    vectors = [(f"doc_{i}", response.embeddings[i], metadata[i]) for i in range(len(chunks))]
    
    # Upsert vectors into Pinecone
    try:
        upsert_response = index.upsert(vectors=vectors)
    except Exception as e:
        st.error(f"Error during upsert: {e}")
except Exception as e:
    st.error(f"Error during embedding: {e}")

def search_query(query, top_k=3):
    query_embedding = co.embed(texts=[query], model="embed-english-v2.0").embeddings[0]
    try:
        result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = result['matches']
        results = [(match['id'], match['score'], match.get('metadata', {}).get('text', 'No text available')) for match in matches]
        return results
    except Exception as e:
        st.error(f"Error during query: {e}")
        return []

def generate_response(query, retrieved_texts):
    context = " ".join(text for _, _, text in retrieved_texts)
    try:
        response = co.generate(
            prompt=f"Based on the following information, answer the question: {query}\n\n{context}",
            model="command-xlarge-nightly"
        )
        return response.generations[0].text.strip()
    except Exception as e:
        st.error(f"Error during response generation: {e}")
        return "Sorry, I couldn't generate a response at this time."

# Streamlit app interface
st.title("Business Policy & Operations Chatbot")
st.write("Ask a question related to business policies and operations to get relevant answers.")

# Input text box for user questions
user_question = st.text_input("Enter your question here:")

# File uploader for PDF
uploaded_file = st.file_uploader("Or upload a PDF file:", type="pdf")

if st.button("Submit"):
    if user_question:
        retrieved_texts = search_query(user_question)
        response_text = generate_response(user_question, retrieved_texts)
        st.write("Chatbot Response:", response_text)
    elif uploaded_file:
        try:
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""
            
            # Chunk and index the PDF text
            pdf_chunks = chunk_text(pdf_text, chunk_size)
            pdf_embeddings = co.embed(texts=pdf_chunks, model="embed-english-v2.0")
            pdf_metadata = [{"text": chunk} for chunk in pdf_chunks]
            pdf_vectors = [(f"pdf_doc_{i}", pdf_embeddings.embeddings[i], pdf_metadata[i]) for i in range(len(pdf_chunks))]
            
            # Upsert PDF vectors into Pinecone
            try:
                upsert_response = index.upsert(vectors=pdf_vectors)
                st.write("PDF content indexed successfully.")
            except Exception as e:
                st.error(f"Error during PDF upsert: {e}")
            
            st.write("Extracted Text from PDF:", pdf_text[:2000])  # Display first 2000 characters of extracted text
        except Exception as e:
            st.error(f"Error processing the PDF file: {e}")
    else:
        st.write("Please enter a question or upload a PDF file.")
