import openai
import os, sys
from openai import OpenAI
import torch
import pandas as pd
import numpy as np
from pathlib import Path


key = open("keys/key-openai-yifei-new", "r").read().strip("\n")
client = OpenAI(api_key=key)


def gen_txt(cif_id, cif_dir="../zeo/zeo_data/dac/MOR/output1"):

    # Get cif
    cif = Path(os.path.join(cif_dir, f"{cif_id}.cif")).read_text()

    text = "Unsuccessful generation"

    # Generate text
    try:
        #Make your OpenAI API request here
        prompt = f"Here is a cif file for a zeolite structure. Provide as much reliable information as possible for this material in descriptive sentences.\n {cif} \n **Answer**: "
        
        response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
        
        text = response.choices[0].message.content
        
    except openai.APIError as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass

    return text



def encode_txt(text):

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # The embedding model you want to use
            input=text,
            encoding_format="float"
        )
    except openai.APIError as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass

    # Access the embedding vectors
    embedding = response.data[0].embedding

    return torch.tensor(embedding)