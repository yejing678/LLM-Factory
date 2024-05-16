import os
import json
import pickle
import copy
import librosa
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from collections import Counter
from loguru import logger

MELD_EMOTION_NAMES =['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

Continue_writing_Format = (
    "### [System]: You are an experienced screenwriter who specialises in continuing stories based on existing plots. Now, continue this plot based on the script provided below and make sure that the renewed plot matches the emotion and style of the original story. {input}"
)

EmoRec_Format= (
    "### [System]: You are a professional emotion recognition expert. Now, predict the emotion of the target utterance. DO Not explain! Your answer should only be one of ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']. {input}"
)

_SAMPLE_RATE = 16000


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

class MELD(Dataset):
    def __init__(self, 
                 raw_data_path, # /datasets/ERC/MELD
                 processed_data_path, # /datasets/ERC/IEMOCAP/Processed/train.pkl
                 tokenizer=None, 
                 audio_processor=None, 
                 max_seq_len=1024, # max input sequence length
                 input_dialog_ratio=0.5, # ratio of input dialogues
                 nshard=1, # number of shards
                 rank=0, # rank of the shard
                 task_type='continue_writing', # the instruct task type
                 context_window_size=5
                 ):
        self.base_path = raw_data_path
        self.max_length = max_seq_len
        self.nshard = nshard
        self.rank = rank
        self.task_type=task_type
        assert task_type in ['continue_writing', 'emotion_recognition'] # task type
        self.context_window_size=context_window_size # only enable in emotion recognition task
        self.input_dialog_length = input_dialog_ratio
        self.text_tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.data = self.read_data(processed_data_path, emotion_enable = False)

    def read_data(self, processed_data_path, emotion_enable=False):
        with open(processed_data_path, 'rb') as file:
            raw_data = pickle.load(file)
        r_data = copy.deepcopy(raw_data)
        data = []
        dialogues = {}
        
        sorted_data = sorted(r_data.items(), key=lambda x:(int(x[1]['dialogue_id']), int(x[1]['utterance_id'])))
        start, end = get_shard_range(len(sorted_data), self.nshard, self.rank)
        sorted_data = sorted_data[start:end]

        # group the utterances by dialog_id
        for entry_name, sample in sorted_data:
            if sample['dialogue_id'] not in dialogues:
                dialogues[sample['dialogue_id']] = []
            dialogues[sample['dialogue_id']].append(
                {
                    'speaker': sample['speaker'],
                    'text': sample['text'],
                    'emotion': sample['emotion'],
                    'audio_path': sample['audio_path'],
                    'utterance_id': sample['utterance_id'],
                }
            )
        

        if self.task_type=="continue_writing":
            for dialog_id, utterances in dialogues.items():
                input_utterance_length = math.floor(len(utterances)*self.input_dialog_length)
                if not emotion_enable:
                    dialog_history = ' '.join([f"{u['speaker']}: {u['text']} \n" for u in utterances[: input_utterance_length]])
                    reference_response = ' '.join([f"{u['speaker']}: {u['text']} \n" for u in utterances[input_utterance_length:]])
                    emotion_diary = ' '.join([u['emotion'] for u in utterances[: input_utterance_length]])
                    response_emotion_diary = ' '.join([u['emotion'] for u in utterances[input_utterance_length:]])

                else:
                    dialog_history = ' '.join([f"{u['speaker']}: {u['text']} [{u['emotion']}] \n" for u in utterances[: input_utterance_length]])
                    reference_response = ' '.join([f"{u['speaker']}: {u['text']} [{u['emotion']}] \n" for u in utterances[input_utterance_length:]])
                    emotion_diary = ' '.join([u['emotion'] for u in utterances[: input_utterance_length]])
                    response_emotion_diary = ' '.join([u['emotion'] for u in utterances[input_utterance_length:]])
                
                inputs = f"### [Human]: {dialog_history} \n### [Bot]: "
                audio_paths = [u['audio_path'] for u in utterances[: input_utterance_length]]
                sample = {
                    'inputs_text': inputs,
                    'response': reference_response,
                    'input_emotion_diary': emotion_diary,
                    'response_emotion_diary': response_emotion_diary,
                    'audio_paths': audio_paths
                }
                data.append(sample)

        elif self.task_type=="emotion_recognition":
            # given the [i-n:i] history utterance, predict the emotion of the i-th target utterance
            for dialog_id, utterances in dialogues.items():
                for i in range(len(utterances)):
                    if i < self.context_window_size:
                        dialog_history = ' '.join([f"{u['speaker']}: {u['text']}\n" for u in utterances[: i]])
                    else:
                        dialog_history = ' '.join([f"{u['speaker']}: {u['text']}\n" for u in utterances[i-self.context_window_size: i]])
                    inputs = f"### [Human]: {dialog_history} the target utterance is {utterances[i]['speaker']}: {utterances[i]['text']}\n### [Bot]: "
                    audio_path = utterances[i]['audio_path']
                    sample = {
                        'inputs_text': inputs,
                        'response': utterances[i]['emotion'],
                        'audio_paths': audio_path
                    }
                    data.append(sample)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return_dict = {}
        sample = self.data[index]
        audio_paths = sample['audio_paths']
        input_text = sample['inputs_text']
        response = sample['response']
        if self.task_type=="continue_writing":
            return_dict['input_text'] = Continue_writing_Format.format(input=input_text)
            return_dict['input_ids'] = self.text_tokenizer(Continue_writing_Format.format(input=input_text), add_special_tokens=True).input_ids
        elif self.task_type=="emotion_recognition":
            return_dict['input_text'] = EmoRec_Format.format(input=input_text)
            return_dict['input_ids'] = self.text_tokenizer(EmoRec_Format.format(input=input_text), add_special_tokens=True).input_ids
        return_dict['attention_mask'] = [1] * len(return_dict['input_ids'])

        return_dict['response'] = response
        return_dict['wav_paths'] = audio_paths
        return return_dict
