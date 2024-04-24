import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import nest_asyncio
from llama_index.core.embeddings import resolve_embed_model

# Fix for running asyncio in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID_MISTRAL")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY_MISTRAL")

# Setup Elasticsearch connection
es_vector_store = ElasticsearchStore(
    index_name="calls",
    vector_field='conversation_vector',
    text_field='conversation',
    es_cloud_id=ELASTIC_CLOUD_ID,
    es_api_key=ELASTIC_API_KEY
)

# Configure the embedding and LLM settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.llm = Ollama(model="llama3:instruct", request_timeout=60.0)

index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(Settings.llm, similarity_top_k=10)

st.title('Document Query Interface')

def process_and_query_documents(query):
    if query:
        try:
            query_embedding = Settings.embed_model.get_query_embedding(query)
            bundle = QueryBundle(query, embedding=query_embedding)
            response = query_engine.query(bundle)
            if hasattr(response, 'response'):
                return response.response  # Assuming 'response' contains the text or relevant output
            else:
                # Additional debug information if no 'response' attribute
                st.write("Check response attributes:", response)
                return 'No results found or response format is different.'
        except Exception as e:
            return f"Error during query execution: {str(e)}"
    else:
        return 'Please enter a query.'

query = st.text_input("Enter your query here:", placeholder="Type your query and press enter...")
if st.button("Search"):
    results = process_and_query_documents(query)
    st.write(results)
