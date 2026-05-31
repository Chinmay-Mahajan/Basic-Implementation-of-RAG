from pydantic import BaseModel 
from fastapi import FastAPI 


from main import RAGpipeline 
from main import QueryRewriter
from main import run_pipeline 



app = FastAPI() 

class QueryRequest(BaseModel):
    query: str 

ra = RAGpipeline(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2" ,dir_path="RAG_API/notes",
              chunk_size=200 , chunk_overlap=20 , show_node=True, show_mem=True )

qw = QueryRewriter() 

@app.post('/query')
def query_endpoint(req : QueryRequest):
    query = req.query 
    op = run_pipeline(ra , qw , query)

    return op

