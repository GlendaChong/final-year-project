import json
import logging 
import os
import requests

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logging.basicConfig(
    level=logging.INFO,  # Capture INFO and above levels
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# # Doesn't work because Dataset scripts no longer supported
# # ds = load_dataset("ronaldahmed/scitechnews") 
# # Thus, have to convert to parquet files
# def convert_json_to_parquet(old_path, new_path): 
#     df = pd.read_json(old_path, lines=True)
#     df.to_parquet(new_path)

# # Load SciTechNews dataset as a parquet file
# df = pd.read_parquet("scitechnews_dataset/valid.parquet")


# Load subsets of datasets 
def load_datasets():
    subset_names = ['computer_science', 'medicine', 'biology']
    datasets = {}
    for subset_name in subset_names:
        datasets[subset_name] = json.load(open(f'datasets/{subset_name}_.json'))
    
    cs_df = pd.DataFrame(datasets['computer_science'])
    med_df = pd.DataFrame(datasets['medicine'])
    bio_df = pd.DataFrame(datasets['biology'])
    
    return cs_df, med_df, bio_df



# Randomly sample from dataset
def random_sampled_dataset(df, sample_size): 
    sampled_ds = df.sample(sample_size, random_state=42)
    return sampled_ds



def add_introduction_column(row):
    # Find index of section name containing 'introduction'
    for i, name in enumerate(row['sc-section_names']):
        if 'introduction' in name.lower():
            return row['sc-sections'][i]
    return None  # No introduction section found


def add_conclusion_column(row): 
    # Find index of section name containing 'conclusion'
    for i, name in enumerate(row['sc-section_names'][1:]):
        if 'conclusion' in name.lower():
            return row['sc-sections'][i]
    return None  


def preprocess_data(df): 
    # Separate introduction section
    df['sc-introduction'] = df.apply(add_introduction_column, axis=1)
    df['sc-conclusion'] = df.apply(add_conclusion_column, axis=1)
    return df
    

# Extract major claim from scientific paper text
def extract_major_claim(text, client): 
    prompt_to_retrieval = (
        "Read the following scientific paper, and identify the main topic of the paper."
        f"Scientific Paper: {text}\n\n"
        "Give a few keywords of the main topic."
    )

    response = client.responses.create(
        model="gpt-4o",
        input=prompt_to_retrieval,
        temperature=0.0,  # deterministic output
    )

    logging.info(f"Major claim extraction response: {response.output_text}")
    return response.output_text


# Retrieve relevant context from web using major claim
def serper_search(claim):
    api_key = os.getenv('SERPER_API_KEY') 
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    query = f"{claim}. surveys statistics expert opinions societal impact. not research papers of the claim itself."
    payload = {
        "q": query
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  
    
    data = response.json()
    results = data.get('organic', []) # 'organic' key usually contains the main search results list
    print(results)
    return results


# Create prompt for news report generation
def create_prompt(row): 
    prompt = (
        f"Title: {row['sc-title']}\n\n"
        f"Abstract: {row['sc-abstract']}\n\n"
        f"Introduction: {row['sc-introduction']}\n\n"
        f"Conclusion: {row['sc-conclusion']}\n\n"
        f"Authors: {row['sc-authors']}\n\n" 
        f"Additional Context: {row['retrieved_context']}\n\n"
        f"Write a clear and engaging news report for the general public with non-technical background based on the above scientific abstract, introduction, conclusion, authors and other additional context obtained from web search."
    )

    return prompt


# Generate news report using GPT-4o model
def generate_news_report_gpt(prompt, client): 
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
    )

    return response.output_text


# LLM as a Judge
def evaluate_generated_output_judge(generated_text, reference_text, client):
    prompt = (
        "You are a reviewer scoring a news report generated from scientific content.\n"
        "Evaluate the news report on these criteria from 1 (poor) to 5 (excellent):\n"
        "1. Factuality: Is the information accurate according to the reference?\n"
        "2. Relevance: Does the news cover the main points clearly?\n"
        "3. Attractiveness: Is the writing engaging for a general audience?\n\n"
        f"Reference news report: {reference_text}\n"
        f"Generated news report: {generated_text}\n"
        "Respond ONLY with a raw JSON object with keys 'factuality', 'relevance', 'attractiveness' and numeric values from 1 to 5. Do not include add code blocks, or markdown."
    )

    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.0,  # deterministic scoring
    )

    logging.info(f"Judge response: {response.output_text}")
    scores = json.loads(response.output_text)  # Parse JSON response for numeric scores
    return scores


def evaluate_and_expand(row, client):
    scores = evaluate_generated_output_judge(row['generated_news'], row['sc-article'], client)
    return pd.Series(scores)  # expands dictionary to columns



def main():
    client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

    # Load datasets
    cs_df, med_df, bio_df = load_datasets()
    df = pd.concat([cs_df, med_df, bio_df], ignore_index=True)


    # sample_size = 100
    # sampled_ds = random_sampled_dataset(sample_size)
    # preprocessed_ds = preprocess_data(sampled_ds)


    # # Retrieval Augmentation
    # preprocessed_ds['major_claim'] = preprocessed_ds['sc-article'].apply(lambda text: extract_major_claim(text, client))
    # preprocessed_ds['retrieved_context'] = preprocessed_ds['major_claim'].apply(lambda claim: serper_search(claim))

    # # Prompting 
    # preprocessed_ds['prompt'] = preprocessed_ds.apply(create_prompt, axis=1)
    # preprocessed_ds['generated_news'] = preprocessed_ds['prompt'].apply(lambda row: generate_news_report_gpt(row, client))

    # # Evaluation
    # evaluation_scores = preprocessed_ds.apply(lambda row: evaluate_and_expand(row, client), axis=1)
    # preprocessed_ds = pd.concat([preprocessed_ds, evaluation_scores], axis=1)

    # preprocessed_ds.to_parquet('scitechnews_dataset/preprocessed_with_rag.parquet')



if __name__ == "__main__": 
    main()
    # df = pd.read_parquet("scitechnews_dataset/preprocessed_with_rag.parquet")







