import os
import time
import logging
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


#generating ids for documents

def generate_doc_id(docs,chunk_index):
    source = docs.metadata.get("source","")
    content = docs.page_content
    hash_input = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{source}:{chunk_index}:{hash_input}"


#logging

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
#Index
load_dotenv()
tavily_client = os.environ.get("TAVILY_API_KEY")



def source_url(documents):
    sources = []
    seen = set()

    for doc in documents:
        source = doc.metadata.get("source")

        if source and source not in seen:
            seen.add(source)
            sources.append(source)   

    return sources

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load Docs
logger.info("Loading %d source documents...", len(urls))
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
logger.info("Loaded %d documents.", len(docs_list))


#Spliter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000,chunk_overlap = 150
)

docs_splits = text_splitter.split_documents(docs_list)

docs_ids = [
    generate_doc_id(doc,i)
    for i,doc in enumerate(docs_splits)
]

logger.info(f"Created {len(docs_splits)} chunks. Building vector store with Ollama...")

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
 
logger.info("Warming up embedding model 'nomic-embed-text' (first call may be slow)...")
t0 = time.time()
embedding_model.embed_query("warmup")
logger.info("Embedding model ready in %.2fs.", time.time() - t0)
 
# reflects real embedding throughput, not cold-start load time.
logger.info("Building vector store with Ollama (%d chunks to embed)...", len(docs_splits))

t0 = time.time()

CHROMA_DIR = "./chroma_db"
vectorstore = Chroma(
    collection_name="adaptive-rag",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR
)



existing_data = vectorstore.get()

existing_ids = set(existing_data["ids"])

logger.info(
    "Existing documents in Chroma: %d",
    len(existing_ids)
)

new_docs = []
new_ids = []

for doc, doc_id in zip(docs_splits, docs_ids):
    if doc_id not in existing_ids:
        new_docs.append(doc)
        new_ids.append(doc_id)

logger.info(
    "New chunks to ingest: %d/%d",
    len(new_docs),
    len(docs_splits)
)
batch_size = 8
t0 = time.time()
if new_docs:
    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i:i + batch_size]
        batch_ids = new_ids[i:i + batch_size]

        vectorstore.add_documents(
            batch,
            ids=batch_ids
        )

        logger.info(
            "Embedded %d/%d new chunks (%.1fs elapsed)",
            min(i + batch_size, len(new_docs)),
            len(new_docs),
            time.time() - t0,
        )
else:
    logger.info("No new chunks to ingest.")



logger.info("Vector store built in %.2fs total.", time.time() - t0)



retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

logger.info("Retriever ready: %s", retriever)

 
logger.info("Vector store ready. Grading retrieved documents...")

logger.info(
    "Ingestion completed in %.2fs",
    time.time() - t0
)

logger.info(
    "Chroma database stored at: %s",
    CHROMA_DIR
)