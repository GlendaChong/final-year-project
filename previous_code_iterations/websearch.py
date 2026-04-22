import requests
import os
from dotenv import load_dotenv


load_dotenv()

def serper_search(query):
    api_key = os.getenv('SERPER_API_KEY')  
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  
    
    data = response.json()
    results = data.get('organic', [])
    return results


results = serper_search("optical phased array")
for r in results[:3]:
    print(f"Title: {r.get('title')}")
    print(f"Link: {r.get('link')}")
    print(f"Snippet: {r.get('snippet')}\n")
