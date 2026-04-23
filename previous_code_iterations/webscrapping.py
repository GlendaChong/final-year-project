import requests
from bs4 import BeautifulSoup

def scrape_metadata(url): 
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the abstract section from meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    abstract = meta_desc["content"] if meta_desc else "Abstract not found."

    # Extract the authors from meta dc.creator tags
    authors = []
    for meta in soup.find_all("meta", attrs={"name": "dc.creator"}):
        if "content" in meta.attrs:
            authors.append(meta["content"])
    
    return abstract, authors

# url = "https://www.nature.com/articles/s43856-021-00042-y"
url = "https://www.nature.com/articles/s41477-022-01202-1"
abstract, authors = scrape_metadata(url)

print("ABSTRACT:")
print(abstract)
print("\nAUTHORS:")
for author in authors:
    print(author)