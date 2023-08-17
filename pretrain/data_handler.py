import os
import re, json
from tqdm import tqdm
import pandas as pd
from itertools import chain

import underthesea
import py_vncorenlp

import datasets

rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"], save_dir="RWKV/model/vncorenlp"
)


def text_normalize(text: str, tonal_symbol_norm=True):
    text = re.sub(r"\s\s+", " ", text)
    if tonal_symbol_norm:
        text = underthesea.text_normalize(text)
    text = " ".join(rdrsegmenter.word_segment(text))
    return text


def load_articles_from_folder(
    folder_path: str, file_type="list_text"
) -> datasets.Dataset:
    if file_type == "article":
        ls_articles = []
        for file_name in tqdm(os.listdir(folder_path), desc="Loading data"):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    raw_data = json.load(f)
                    article = raw_data.get("meta", {})
                    text = {
                        "text": article.get("title", "")
                        + " "
                        + article.get("snippet", "")
                        + " "
                        + article.get("message", "")
                    }
                    ls_articles.append(text)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=ls_articles))
        return dataset

    elif file_type == "list_text":
        ls_samples = []
        for file_name in tqdm(os.listdir(folder_path), desc="Loading data"):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    raw_data = json.load(f)
                    for text in raw_data:
                        if isinstance(text, str):
                            text = {"text": text}
                            ls_samples.append(text)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=ls_samples))
        return dataset

    else:
        return None


def load_preprocessed_dataset(
    folder_path, tokenizer, train_ratio=0.95, context_length=1024, file_type="list_text"
):
    raw_dataset = load_articles_from_folder(
        folder_path=folder_path, file_type=file_type
    )
    train_data, val_data = raw_dataset.train_test_split(train_size=train_ratio).values()
    dataset = datasets.DatasetDict({"train": train_data, "validation": val_data})

    column_names = list(dataset["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = dataset.map(
        lambda dataset: tokenizer(dataset[text_column_name]),
        batched=True,
        remove_columns=column_names,
    )

    def group_texts(examples, block_size=context_length):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    return lm_datasets
