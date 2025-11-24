import os, sys

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline

import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# os.environ["HUGGING_FACE_HUB_TOKEN"] = YOUR_CODE

def clear_cache():
  if torch.cuda.is_available():
    model = None
    torch.cuda.empty_cache()

clear_cache()
def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

device


TEMP = 0.01
MAX_NEW_TOKENS = 1000 #change accordingly

MODEL_NAME    = 'aleynabeste/ZeoDapModelLR1e5'
MODEL_ID  =  "zeo_dapt_llama"
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID  =  "llama_8B_instruct"


def answer_llama(question):


    prompt_text = f""" <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
    {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    raw_answer = llm.invoke(prompt_text) 
    return raw_answer

def gen_txt(cif_id, cif_dir="cif_files/MOR"):
    cif = Path(os.path.join(cif_dir, f"{cif_id}.cif")).read_text()

    question = f"Here is a cif file for a zeolite structure. Provide as much reliable information as possible for this material in descriptive sentences.\n {cif} \n **Answer**: "

    text = answer_llama(question)

    response = text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]

    return response

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16, 
        trust_remote_code=True, 
        device_map=device
    )

    model.tie_weights()


    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = MAX_NEW_TOKENS 
    generation_config.temperature = TEMP

    model.generation_config = generation_config 


    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"device":device})

    CIF_DIR = "cif_files/MOR"

    files = glob.glob(f"{CIF_DIR}/*.cif")

    records = []

    for file in tqdm(files):
        cif_id = os.path.basename(file).split(".cif")[0]
        response = gen_txt(cif_id=cif_id,
                          cif_dir=CIF_DIR)
        records.append(
            {
                "cif_id": cif_id,
                "response": response
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"generated_text/{MODEL_ID}.csv", index=False)