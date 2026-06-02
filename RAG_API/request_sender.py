import requests
from typing import List


class Sender():
    def __init__(self , base_url , show_internal_var:bool):
        self.base_url = base_url 
        self.end_points = self._get_endpoint_info()
        self.show_internal_var = show_internal_var

    def _check(self, response:requests.Response):
        return response.ok 
    

    def _get_endpoint_info(self):
        url = f"{self.base_url}/endpoints"
        response = requests.get(url) 
        response = response.json()
        return response



    def send_query_to_url(self ,query:str):
        q = {"query":f"{query}"} 
        url = f"{self.base_url}{self.end_points['query']}" 
        response = requests.post(url , json = q)
        if (self._check(response)):
            # print("Success")
            data = response.json()
            # print("-"*100 + "RESPONSE" + "-"*100)
            print(data['answer'])
            if (self.show_internal_var):
                print("="*50 + "retrieved chunks" +"="*50)
                print(data['"retrieved_chunks"'])
                print("="*50 + "Re-written queries" +"="*50)
                print(data['rewritten_queries'])
            return data['answer']    
        else:
            print("Post unsuccessfull")    
            return None    

    def send_to_refresh_endpoint(self):
        url = f"{self.base_url}{self.end_points['refresh']}"
        response = requests.post(url) 
        if (self._check(response)):
            print("Refresh success , new vector index created")
        else:
            print(f"Refresh not successfull , post not sucessfull to the url {url}")

    def send_to_upload_endpoint(self , file_paths:List[str]):
        url = f"{self.base_url}{self.end_points['upload']}"
        files = [open(path , 'rb') for path in file_paths]
        try:
            payload = [('files' , f) for f in files]
            response = requests.post(url , files=payload)
            data = response.json()
            print(data)
            # print(data["uploaded_files"])
            # print(data['not_uploaded_files'])
        finally:
            for f in files:
                f.close()

       

BASE_URL = "http://127.0.0.1:8000"

