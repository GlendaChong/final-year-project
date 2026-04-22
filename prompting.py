import json
import os
import requests
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI
from transformers import pipeline


load_dotenv()
device = torch.device("cpu")
login(token=os.getenv("HF_TOKEN"))

# Load the SciNews dataset
ds = load_dataset("dongqi-me/SciNews") 

# Randomly sample from dataset
def random_sampled_dataset(sample_size): 
    sampled_ds = ds["train"].shuffle(seed=42).select(range(sample_size))
    return sampled_ds


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


def get_info_from_paper(paper_body): 
    # Extract Introduction 
    # Extract Conclusion 
    return 

def create_rag_enhanced_prompt(abstract, authors, rag_system): 
    return 



def create_prompt(abstract, authors): 
    authors_joined = ", ".join(authors)
    # prompt = (
    #     f"Write a clear and engaging news report for the general public with non-technical background based on the following scientific abstract and author list:\n\n"
    #     f"Abstract: {abstract}\n\n"
    #     f"Authors: {authors_joined}\n"
    # )

    prompt = (
        "Your task is to write a clear, concise, and engaging news article for a general audience, based on the scientific abstract and author list.\n"
        "• Avoid technical jargon—explain or omit complex terms.\n"
        f"\n---\nABSTRACT:\n{abstract}\n\nAUTHORS: {authors_joined}\n---\n\n"
        "Now write the news article."
    )

    return prompt


# Generate news report using GPT-4o model
def generate_news_report_gpt(prompt, client): 
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
    )

    return response.output_text


# Generate news report using LLaMA model
def generate_news_report_llama(prompt, generator):
    response = generator(prompt, max_new_tokens=30)
    # print(response)
    return response[0]['generated_text']


def main(): 
    sample_size = 5
    sampled_ds = random_sampled_dataset(sample_size)
    client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
    generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)

    # Extract generated news reports
    combined_data = []

    for record in sampled_ds: 
        paper_url = record["Paper_URL"]
        if paper_url: 
            abstract, authors = scrape_metadata(paper_url)

        prompt = create_prompt(abstract, authors)
        news_gpt = generate_news_report_gpt(prompt, client)
        # news_llama = generate_news_report_llama(prompt, generator)

        combined_data.append({
            "paper_url": paper_url, 
            "abstract": abstract, 
            "authors": authors, 
            "generated_news_body_gpt": news_gpt, 
            # "generated_news_body_llama": news_llama, 
            "news_body": record["News_Body"],
            "paper_body": record["Paper_Body"]
        })

    # Save as json file
    version = 8
    with open(f"generated_news_reports/generated_news_reports_{version}.json", "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__": 
    main()


