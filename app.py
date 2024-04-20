import gradio as gr
import os
from dotenv import load_dotenv
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, QueryBundle, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import resolve_embed_model
import nest_asyncio

# Apply asyncio fix
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

# Configure LLM and embedding model
Settings.llm = Ollama(model="llama3:instruct", request_timeout=60.0)
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# Create a query engine from the vector store
index = VectorStoreIndex.from_vector_store(es_vector_store)
query_engine = index.as_query_engine(Settings.llm, similarity_top_k=10)

def query_documents(query):
    """ Query the document database based on the input query and return the responses. """
    if query:
        try:
            bundle =  QueryBundle(query, embedding=Settings.embed_model.get_query_embedding(query))
            response = query_engine.query(bundle)
            if response.responses:
                return '\n'.join([resp.text for resp in response.responses])
            else:
                return 'No results found.'
        except Exception as e:
            return f"Error during query execution: {str(e)}"
    else:
        return 'Please enter a query.'

# Set up the Gradio interface
iface = gr.Interface(
    fn=query_documents,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Textbox(label="Query Results"),
    title="Document Query Interface",
    description="Enter a query to search through the document database."
)

if __name__ == "__main__":
    iface.launch()
