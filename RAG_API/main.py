from llama_index.core import VectorStoreIndex 
from llama_index.core import SimpleDirectoryReader 
from llama_index.llms.ollama import Ollama 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage , MessageRole
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.retrievers.bm25 import BM25Retriever
import json 
import numpy as np 
import os 
import shutil

from prompts import qw_raw
from prompts import qa_prompt_tmpl
from ingestion import Ingestor


qw_prompt = PromptTemplate(qw_raw)

qa_prompt=PromptTemplate(qa_prompt_tmpl)


def load_data(input_path):
    '''

    Load files from file directory.
    Automatically select the best file reader given file extensions.
    '''
    reader = SimpleDirectoryReader(input_dir = input_path , exclude_empty=True) 
    data = reader.load_data()
    text = [] 
    for doc in data:
        text.append(doc.get_content())
    return data


def load_emb_model(model_name):
    '''
    loading the embedding model to convert chunks -> vector embeddings , for retrieval 
    '''
    emb = HuggingFaceEmbedding(model_name=model_name)
    return emb

def load_llm():
    '''
    loading a local LLM (in my case llama3 - 8b)
    '''
    llm = Ollama(model="llama3" , request_timeout=120.0)
    return llm 

def cosine_sim(x , y):
    z = np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return z

def check_dir(dir):
    if os.path.isdir(dir):
        return True
    return False

class RAGpipeline():
    '''
    Arguments:
    1. embedding_model_name - the name of the embedding model to be used
    2. dir_path - the path of the directory where text files are stored
    3. chunk_size - the size of the text (in tokens) which is embedded into a vector 
    4. chunk_overlap - the size of the overlap between chunks (in tokens)
    5. verbose - (bool) if true shows the retrieved nodes
    
    '''
    def __init__(self , embedding_model_name , dir_path , chunk_size=200, chunk_overlap = 20 , show_node=True , show_mem = False ):
        
        self.llm = load_llm()
        self.emb_model = load_emb_model(model_name=embedding_model_name)
        self.dir_path = dir_path 
        self.ingestor = Ingestor(self.dir_path , chunk_overlap=chunk_overlap , chunk_size=chunk_size)
        if os.path.exists("./storage/docstore.json"):
            storage_context = StorageContext.from_defaults(persist_dir="./storage") # if the storage dir does exist then we make the vector index using the stored index. We dont do chunking , embedding here we just load up the stored vector embeddings.
            self.index = load_index_from_storage(storage_context,embed_model=self.emb_model)

        else:   
            # if the path doesn't exist then we must do loading-chunking - embedding 
            self.ingestor.save_emb() # saves embedding for future uses.
            self.documents = load_data(self.dir_path)
            self.splitter = SentenceSplitter(chunk_overlap=chunk_overlap , chunk_size=chunk_size) # we use this to split the documents into chunks
            self.index = VectorStoreIndex.from_documents(self.documents , embed_model = self.emb_model , transformations=[self.splitter])

            # performs chunking , embedding , and stores vectors in such a way that similarity searches are efficient
        self.retriver = self.index.as_retriever(similarity_top_k = 10) # returns a retriever object by which we can retrieve the top_k number of chunks given a input query based on similarity
        self.show_node = show_node
        self.show_mem = show_mem # shows chat memory after each assitant user exchange

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000) # forms a memory buffer for chat conversations , takes into account the possiblility of token overflow. Stores the last k tokens such that they fit in the token limit.
        self.chat_engine = self.index.as_chat_engine(llm = self.llm , text_qa_template = qa_prompt , chat_mode="context" , memory = self.memory) #This actually performs another RAG pipeline by itself , I initally didnt know about this and used it as is.
        #later when i inspected the chat history , I found two copies of each conversation in it. This will affect the overall performance of the RAG pipeline. So I chose to manually implement core procedures for RAG.
        self.prompt = qa_prompt # this is the prompt template I will use to format the user query - This was made by CHATGPT to help me lay out the rules in a clear manner.

    def gen_response(self , query , docs):
        # resp = self.chat_engine.chat(query)
        history = "\n".join(f"{m.role}: {m.content}" for m in self.memory.get_all()) # get the chat memory till now
        resp = self.llm.complete(self.prompt.format(context_str=docs , query_str = query , ChatHistory=history)) #Pass the history to the prompt (we want the LLM to be able to look at the chat history) and also retrived documents.
        self.memory.put(ChatMessage(role=MessageRole.USER , content=query)) # Update the memory with the user's query
        self.memory.put(ChatMessage(role=MessageRole.ASSISTANT , content=resp.text)) #Update the memory with the LLM's Answer
        return resp


    def mannual_rerank(self, rewritten_query , nodes , top_k):
        '''
        Using Cosine similarity filter out top_k chunks from a larger number of chunks.
        Rerank chunks for one query
        '''

        score = []
        qv = self.emb_model.get_text_embedding(rewritten_query) 
        for node in nodes:
            nv = self.emb_model.get_text_embedding(node.text)
            score.append((cosine_sim(qv , nv) , node))
        score.sort(reverse=True , key= lambda x:x[0])
        best_nodes = [node for _ , node in score[:top_k]]
        return best_nodes



    def show_nodes(self , query):
        nodes = self.retriver.retrieve(query) # searches the VectorStoreIndex to find nearest negihbours (or most similary top_k) 
        # returns a list of [NodeWithScore] objects , can do node.score to get the similarity score.
        print(f"Showing the retrieved chunks (nodes) for the query {query}")
        for i , node in enumerate(nodes):
            print(f"chunk {i+1}")
            print(f"Node score {node.score}")
            print(node.text)
            print("-"*100)
        print("-"*100)   


    def refresh_index(self):
        if os.path.exists("./storage"): # if ./storage exists then remove it 
            shutil.rmtree("./storage")
        self.ingestor = Ingestor(self.dir_path,chunk_overlap=20,chunk_size=200)  #make another ingestor object (because the ingestor object has 
        #this ---> self.documents = SimpleDirectoryReader(input_dir=input_dir , exclude_empty=True).load_data()) , meaning it read the dir when it was initalised , hence we have to make another object to overwrite the old one
        self.ingestor.save_emb() # save the embeddings to the local storage
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        self.index = load_index_from_storage(storage_context,embed_model=self.emb_model)
        self.retriver = self.index.as_retriever(similarity_top_k=10) # reset the retriver too , (SAME ISSUE AS THE BUG PREVIOUSLY FOUND OUT)



class QueryRewriter():
    '''
    
    Directly trying to search the vector index using users query can lead to sub-optimal results as the query itself may not be properly worded , perhaps it could have some unnessesary info. That could cause low similarity score.
    Hence another LLM is used to :
    Reduce users queries to it's most basic form , stripping it of puntuations and other not needed things.
    

    '''
    def __init__(self ):
        self.llm = load_llm()
        # self.emb_model = load_emb_model(model_name=embedding_model_name)
        self.prompt = qw_prompt # query rewriter prompt template
        
    def rewrite_query(self , user_query , chat_history):
        # passing it chat_history as the QueryRewriter must be able to figure of in the query what does "it" or "they" etc point to.
        history_text = "\n".join([f"{msg.role}:{msg.content}" for msg in chat_history])
        prompt = self.prompt.format(chat_history=history_text , user_message = user_query)
        resp = self.llm.complete(prompt=prompt).text
        print(resp)
        for i in range(3): # try to re-run the llm with the same prompt if the output is not json 
            resp = self.llm.complete(prompt=prompt).text
            try:
                queries = json.loads(resp)["queries"]
                return queries
            except Exception as e:
                print(f"Error occurred trying again attempt {i+1} , {e} ")
        return user_query        

def run_pipeline(ra , qw , query):
    """
    Runs a complete Retrieval-Augmented Generation (RAG) inference cycle.

    The user query is rewritten into retrieval-friendly search queries,
    relevant chunks are retrieved and reranked, and the selected context
    is passed to the LLM to generate the final answer.

    Returns the generated response together with the rewritten queries
    and retrieved context chunks for debugging and evaluation.
    """
 
    chat_history = ra.memory.get_all()
    queries = qw.rewrite_query(query , chat_history)
        
    docs = "\n"
    retrieved_chunks =[]
    for q in queries:
        nodes = ra.retriver.retrieve(q)
        best_nodes = ra.mannual_rerank(q , nodes , top_k=3) # reranking mannually using cosine similarity
        for node in best_nodes:
            docs += "\n" + node.text
            retrieved_chunks.append(node.text)
        
    # prompt = ra.prompt.format(context_str = docs , query_str=query)
    resp = ra.gen_response(query=query , docs = docs)
    return {
        "answer":resp.text,
        "rewritten_queries":queries,
        "retrieved_chunks": retrieved_chunks
    }
    # resp = ra.gen_response(resp)
    # print(f"{'-'*50}+RESPONSE+{'-'*50}")
    
    # print(resp)
    # print("-"*100)
    # if (ra.show_node):
        
    #     print("Showing the nodes....")
    #     for q in queries:
    #         ra.show_nodes(q)
    # if (ra.show_mem):
    #     print("Showing memory")
    #     print(ra.memory.get_all())     



def run_cli(ra , qw):
    while True:
        query = input("ENTER PROMPT: ")
        if query == "quit":
            break
        result = run_pipeline(query, ra, qw)
        if ra.show_node:
            print(result['retrieved_chunks'])
        if ra.show_mem:
            print(ra.memory.get_all())    
        print(result["answer"])

