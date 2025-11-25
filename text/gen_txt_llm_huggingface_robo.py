import os, sys

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline

import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# os.environ["HUGGING_FACE_HUB_TOKEN"] = YOUR_TOKEN

def clear_cache():
  if torch.cuda.is_available():
    model = None
    torch.cuda.empty_cache()

clear_cache()
def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device_map()

device

BASE_DF_PATH = "robo_200_2000.csv"

TEMP = 0.01
MAX_NEW_TOKENS = 1000 #change accordingly

MODEL_NAME    = 'aleynabeste/ZeoDapModelLR1e5'
MODEL_ID  =  "zeo_dapt_llama"
BASE_TXT_ID = "inputrobo"
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID  =  "llama_8B_instruct"


def answer_llama(question):


    prompt_text = f""" <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    
    {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    raw_answer = llm.invoke(prompt_text) 
    return raw_answer

def gen_txt(cif_id, base_text):

    question = f"Here is the structure description for a zeolite structure coded as {cif_id}. Provide as much reliable information as possible for this material in descriptive sentences.\n {base_text} \n **Answer**: "

    text = answer_llama(question)

    response = text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]

    return response.lstrip("\n")

# TODO: Run this
def split_df(all_txt_df_path, txt_dir, llm="llama-3-8B-instruct"):
   all_txt_df = pd.read_csv(all_txt_df_path)
   for start_id in range(0, 2000, 200):
        robo_df_dir = os.path.join(txt_dir, f"zeoDAC_robo_start_{start_id}_sample_200")
        robo_df = pd.read_csv(os.path.join(robo_df_dir, f"robo_{start_id}_None_skip_none.csv"))

        split_jid = list(robo_df["jid"])

        split_df = all_txt_df[(all_txt_df["cif_if"].isin(split_jid))]

        save_dir = f"zeoDAC_{llm}_start_{start_id}_sample_200"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        split_df.to_csv(os.path.join(save_dir, f"{llm}_{start_id}.csv"), index=False)

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

    base_df = pd.read_csv(BASE_DF_PATH)
    records = []

    for row in tqdm(base_df.itertuples(), total=len(base_df)):
        cif_id = row.jid
        base_text = row.text
        response = gen_txt(cif_id=cif_id,
                          base_text=base_text)
        records.append(
            {
                "cif_id": cif_id,
                "response": response
            }
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"generated_text/{MODEL_ID}_{BASE_TXT_ID}.csv", index=False)