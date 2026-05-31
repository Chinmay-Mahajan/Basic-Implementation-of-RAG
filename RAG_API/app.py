from pydantic import BaseModel 
from fastapi import FastAPI 
from fastapi import UploadFile 
from fastapi import File 
import shutil
from typing import List
import os
import tempfile
from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader , VectorStoreIndex

from main import RAGpipeline 
from main import QueryRewriter
from main import run_pipeline 

allowed = [".txt"]
PERSIT_DIR = "./storage"


app = FastAPI() 

class QueryRequest(BaseModel):
    query: str 

ra = RAGpipeline(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2" ,dir_path="notes",
              chunk_size=200 , chunk_overlap=20 , show_node=True, show_mem=True )

qw = QueryRewriter() 

@app.post('/query')
def query_endpoint(req : QueryRequest):
    query = req.query 
    op = run_pipeline(ra , qw , query)

    return op

@app.post('/upload')
def upload_files(files: List[UploadFile] = File(...)):
    #  make a temp directory to store the new files , then use the simple directory reader to read the new temp directory , then merge the new index with the old index
    uploaded_files = []
    not_uploaded_files = []
    with tempfile.TemporaryDirectory() as temp_dir: # makes a temp dirr , with keyword ensures after the code inside this ident is finished the generated dir is deleted 
        # also the generated dirr is randomly named (which ensures two different files dont have acess to eachothers data)

        for file in files:
            if (os.path.splitext(file.filename)[1] not in allowed):
                not_uploaded_files.append(file.filename)
                continue
            # saving the file in BOTH the dir_path and temp_dir (ie. the notes folder and temperoray directory)
            actual_file_path = os.path.join(ra.dir_path , file.filename)
            temp_file_path = os.path.join(temp_dir , file.filename)

            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                uploaded_files.append(file.filename)
            # FACED A BUG HERE , the shutil.copyfileobj streams the files as bytes , and when it's done it points to the EOF (end of file)
            # So when running the open(....) again for saving to the notes folder we end up saving nothing.    
            file.file.seek(0) # makes sure we point to the start of the file
            with open(actual_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)    

        if uploaded_files:
            reader = SimpleDirectoryReader(input_dir=temp_dir) # reading the new documents
            new_documents = reader.load_data()
            if os.path.exists(PERSIT_DIR) and os.listdir(PERSIT_DIR):
                for doc in new_documents:
                    ra.index.insert(doc)
                ra.retriver = ra.index.as_retriever(similarity_top_k=10)    # had to refresh the retriver , as without it the retriver had no clue that the new embeddings were added.
            else:
                # we dont require this else block because ra is already made at the top (in it;s init method we have called ingestor and called saved embedding)
                index = VectorStoreIndex.from_documents(new_documents , embed_model=ra.emb_model , transformations=[ra.splitter])

            ra.index.storage_context.persist(persist_dir=PERSIT_DIR)
            status = "1"
        else:
            status = "0" # return a status of "0" when none of the files were of the allowed extention.

    return {"uploaded_files":uploaded_files , 
            "not_uploaded_files":not_uploaded_files,
            "status":status
            }


@app.post('/refresh')
def refresh():
    ra.refresh_index()
    return {
        "status":"success"
    }





        


