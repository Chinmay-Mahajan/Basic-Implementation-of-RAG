'''
PLAN TO EXTEND THE CURRENT PROJECT --> we already have an app with endpoints like query , upload_file , refresh . What I would like too do is 
Basically have another llm be able to command tools like peform_rag(query) by executing a function that sends a request to the app's query endpoint , and getting the result
'''

from request_sender import Sender
from request_sender import BASE_URL






class ToolManager():
    def __init__(self ):
        self.sender = Sender(BASE_URL , show_internal_var=False)


tm = ToolManager()

def query_RAG_pipeline(query):
    resp = tm.sender.send_query_to_url(query)
    return resp 

def gettemp(city):
    return f"the temp in {city} is 40C"



