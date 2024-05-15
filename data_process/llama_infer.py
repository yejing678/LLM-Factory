import os
import sys
import json
import fire
from tqdm import tqdm
from typing import Dict, List

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from transformers import LlamaTokenizer, LlamaForCausalLM
try:
    from iemocap import IEMOCAP_6_Ways, IEMOCAP_DataCollator
except:
    from .iemocap import IEMOCAP_6_Ways, IEMOCAP_DataCollator
from loguru import logger


Continue_writing_Format = (
    "### [System]: You are an experienced screenwriter who specialises in continuing stories based on existing plots. Now, continue this plot based on the script provided below and make sure that the renewed plot matches the emotion and style of the original story. {input}"
)

def get_shard_range(tot, nshard, rank):
    # Define a function to calculate the processing range for a shard
    assert rank < nshard and rank >= 0, f"invalid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    
    logger.info(
        f"rank {rank} of {nshard}, processing {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    
    return start, end

def get_dataset(manifest, nshard, rank):
    # get a subset of a dataset based on shard information
    with open(manifest, "r") as f:
        lines = f.readlines()
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        lines = [json.loads(line.strip()) for line in lines]
    dataset = Dataset.from_list(lines)

    return dataset

def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][-len(v):])

    return res


@dataclass
class DataCollator:
    pad_id: int = 0

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        audio = [sample["audio"] for sample in samples]
        response = [sample["response"] for sample in samples]
        response_length = [sample["response_length"] for sample in samples]
        input_text=[sample['input_text'] for sample in samples]
        input_length=[sample['length'] for sample in samples]

        input_ids = collate_tokens(input_ids, 0)
        attention_mask = collate_tokens(attention_mask, 0)

        return {
            'input_text': input_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_length": input_length,
            "audio": audio,
            'response': response,
            'response_length': response_length,
        }
    

def continue_writing(
    model_name_or_path, # name_or_path of the model
    manifest, # path to the manifest file
    lab_dir, # path to the directory to save the lab files
    nshard=8,
    rank=0,
    batch_size=1
):

    logger.add(f'continue-writing.log', format="{time} {level} {message}", level="INFO")
    accelerator = Accelerator()
    logger.info(accelerator.state)
    device = accelerator.device

    dataset = get_dataset(manifest, nshard, rank)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

    def process_dataset(batch):
        batch["input_ids"] = tokenizer(Continue_writing_Format.format(input=batch["inputs"])).input_ids
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        batch["audio"] = batch["wav_paths"]
        batch["length"] = len(batch["input_ids"])
        batch["response"] = batch["responses"]
        batch["response_length"] = len(tokenizer(batch['response']).input_ids)
        batch['input_text'] = Continue_writing_Format.format(input=batch["inputs"])
        return batch
    
    def is_in_length_range(length):
            return length > 100 and length < 1024
    
    dataset = dataset.map(process_dataset)
    dataset = dataset.filter(is_in_length_range, input_columns=["length"])

    logger.info(f"Using model: {os.path.splitext(os.path.basename(model_name_or_path))[0]}")
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)

    data_collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )


    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    model.to(device)
    model.eval()

    split = os.path.splitext(os.path.basename(manifest))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"
    os.makedirs(lab_dir, exist_ok=True)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    with open(lab_path, "w") as f:
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=1024,
                do_sample=True,
                num_beams=1,
                top_p=0.75,
                temperature=0.1,
                num_return_sequences=1,
            )
            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            generated_length = generated_tokens.shape[1]
            output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for audio, response, reference in zip(batch["audio"], output_text, batch["response"]):
                json_string = json.dumps(
                    { 
                        "response": response,
                        "reference": reference,
                        "audio": audio,
                    }
                )
                print(json_string, file=f)
                logger.info(json_string)
            progress_bar.update(1)

    logger.info(f"finished successfully, saved file to {lab_path}")
    

def my_continue_writing(
    model_name_or_path, # name_or_path of the model
    raw_data_path,
    processed_data_path,
    lab_dir, # path to the directory to save the lab files
    nshard=8,
    rank=0,
    batch_size=1
):

    logger.add(f'my_continue-writing.log', format="{time} {level} {message}", level="INFO")
    accelerator = Accelerator()
    logger.info(accelerator.state)
    device = accelerator.device
 
    logger.info(f"Using model: {os.path.splitext(os.path.basename(model_name_or_path))[0]}")
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    
    dataset = IEMOCAP_6_Ways(raw_data_path, processed_data_path, tokenizer=tokenizer, nshard=nshard, rank=rank)
    data_collator = IEMOCAP_DataCollator()
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    model.to(device)
    model.eval()

    split = os.path.splitext(os.path.basename(raw_data_path))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"
    os.makedirs(lab_dir, exist_ok=True)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    with open(lab_path, "w") as f:
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=1024,
                do_sample=True,
                num_beams=1,
                top_p=0.75,
                temperature=0.1,
                num_return_sequences=1,
            )
            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for input_text, audio, response, reference in zip(batch['input_text'], batch["wav_paths"], output_text, batch["response"]):
                json_string = json.dumps(
                    { 
                        "input_text": input_text,
                        "response": response,
                        "reference": reference,
                        "audio": audio,
                    }
                )
                print(json_string, file=f)
                logger.info(json_string)
            progress_bar.update(1)

    logger.info(f"finished successfully, saved file to {lab_path}")


def emotion_recogntion(
    model_name_or_path, # name_or_path of the model
    raw_data_path,
    processed_data_path,
    lab_dir, # path to the directory to save the lab files
    nshard=8,
    rank=0,
    batch_size=1
):

    logger.add(f'emotion_recogntion.log', format="{time} {level} {message}", level="INFO")
    accelerator = Accelerator()
    logger.info(accelerator.state)
    device = accelerator.device
 
    logger.info(f"Using model: {os.path.splitext(os.path.basename(model_name_or_path))[0]}")
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    
    dataset = IEMOCAP_6_Ways(raw_data_path, processed_data_path, tokenizer=tokenizer, nshard=nshard, rank=rank, task_type="emotion_recogntion")
    data_collator = IEMOCAP_DataCollator()
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    model.to(device)
    model.eval()

    split = os.path.splitext(os.path.basename(raw_data_path))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"
    os.makedirs(lab_dir, exist_ok=True)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    with open(lab_path, "w") as f:
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=1024,
                do_sample=True,
                num_beams=1,
                top_p=0.75,
                temperature=0.1,
                num_return_sequences=1,
            )
            input_length = batch["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]
            output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for input_text, audio, response, reference in zip(batch['input_text'], batch["wav_paths"], output_text, batch["response"]):
                json_string = json.dumps(
                    { 
                        "input_text": input_text,
                        "response": response,
                        "reference": reference,
                        "audio": audio,
                    }
                )
                print(json_string, file=f)
                logger.info(json_string)
            progress_bar.update(1)

    logger.info(f"finished successfully, saved file to {lab_path}")


if __name__ == "__main__":
    fire.Fire({
        'continue_writing': continue_writing,
        'my_continue_writing': my_continue_writing,
        'emotion_recogntion': emotion_recogntion,
    })