import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import resolve_embed_model

import nest_asyncio

# Apply asyncio fix for Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID_MISTRAL")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY_MISTRAL")

# Initialize the app
st.title('Document Query Interface')

# Setup Elasticsearch connection
es_vector_store = ElasticsearchStore(
    index_name="calls",
    vector_field='conversation_vector',
    text_field='conversation',
    es_cloud_id=ELASTIC_CLOUD_ID,
    es_api_key=ELASTIC_API_KEY
)

# Configure LLM and embedding model
Settings.llm = Ollama(model="llama3:instruct", request_timeout=60.0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")

# Create a query engine from the vector store
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(Settings.llm, similarity_top_k=10)

# User input for query
query = st.text_input('Enter your query:', value='from the Knowledge base Compare the earnings per share in the fiscal year of 2022 and 2023 for Tesla and Alphabet')

if st.button('Search'):
    if query:
        st.write('Searching for:', query)
        # Create a query bundle with embeddings
        query_embedding = Settings.embed_model.get_query_embedding(query)
        bundle = QueryBundle(query, embedding=query_embedding)
        # Perform the query
        try:
            response = query_engine.query(bundle)
            # Display results
            if response.responses:
                for resp in response.responses:
                    st.write(resp.text)
            else:
                st.write('No results found.')
        except Exception as e:
            st.error(f"Failed to execute query: {str(e)}")
    else:
        st.warning('Please enter a query to search.')
