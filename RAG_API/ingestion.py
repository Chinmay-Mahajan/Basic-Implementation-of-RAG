from llama_index.core import StorageContext 
from llama_index.core import VectorStoreIndex 
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter



class Ingestor():
    def __init__(self , input_dir , chunk_overlap , chunk_size):
        self.input_dir = input_dir 
        self.documents = SimpleDirectoryReader(input_dir=input_dir , exclude_empty=True).load_data()
        self.embedding_model=HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.splitter = SentenceSplitter(chunk_overlap=chunk_overlap , chunk_size=chunk_size)
        

    def save_emb(self):
        self.index = VectorStoreIndex.from_documents(self.documents , embed_model = self.embedding_model , transformations=[self.splitter])
        self.index.storage_context.persist("./storage")

    