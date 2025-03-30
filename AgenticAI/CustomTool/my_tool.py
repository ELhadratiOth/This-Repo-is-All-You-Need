from crewai_tools import BaseTool
import requests
import json
import os


class MyTool(BaseTool):
    name : str = "Serper Dev Tool"
    description : str = "this  tool is  used to  get  news from the  web"
    
    def _run(self, input: str) -> str:
        """
        Search the web for the given news input
        """

        url = "https://google.serper.dev/news"

        payload = json.dumps({
            "q": input ,
            "num": 10,
            "autocorrect":False ,
            "tbs":"qdr:d"
        })
        headers = {
        'X-API-KEY': os.getenv("SERPER_API_KEY"),
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        response_cleaned = response.json().get("news" , [])




        return json.dumps(response_cleaned , indent=2) 