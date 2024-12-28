#!/usr/bin/env python3

import functools
import os
import requests
from functools import lru_cache
from typing import Dict, List
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor
from tabulate import tabulate

import tiktoken
from tiktoken.model import MODEL_TO_ENCODING

__all__ = [
    "compress_rate",
    "estimate_moonshot_tokens",
]


@lru_cache(maxsize=16)
def load_tokenizer(model_name: str):
    if model_name.startswith("sp"):  # local model
        local_path = model_name.split("/")[-1]
        tokenizer = SentencePieceProcessor(local_path)
        return tokenizer
    if model_name in MODEL_TO_ENCODING:  # oai model
        encoding = tiktoken.encoding_for_model(model_name)
        return encoding

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def get_vocab_size(tokenizer) -> int:
    if isinstance(tokenizer, SentencePieceProcessor):
        return tokenizer.get_piece_size()
    if isinstance(tokenizer, tiktoken.Encoding):
        return tokenizer.n_vocab
    return tokenizer.vocab_size  # huggingface tokenzier


def estimate_moonshot_tokens(text, model: str = "moonshot-v1-8k", api_key=None) -> int:
    if api_key is None:
        api_key = os.environ.get("MOONSHOT_API_KEY", "")
    if not api_key:
        print("!!!Warning: api_key is empty.")

    url = "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        num_tokens = response.json()["data"]["total_tokens"]
        return num_tokens
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def compress_rate(text: str, model_name: str, language: str = "en", verbose: bool = True) -> Dict:
    if "moonshot" in model_name:
        f = functools.partial(estimate_moonshot_tokens, model=model_name)
        num_tokens = f(text)
        vocab_size = -1
    else:
        tokenizer = load_tokenizer(model_name)
        if isinstance(tokenizer, tiktoken.Encoding):
            tokens = tokenizer.encode(text)
        else:
            tokens = tokenizer.tokenize(text)
        num_tokens = len(tokens)
        vocab_size = get_vocab_size(tokenizer)

    num_characters = len(text)
    num_words = len(text.split()) if language == "en" else num_characters

    chars_per_token = num_characters / num_tokens
    words_per_token = num_words / num_tokens

    if verbose:
        print(f"number of characters: {num_characters:,d}")
        print(f"number of words: {num_words:,d}")
        print(f"number of tokens: {num_tokens:,d}")
        print(f"token: character -> 1: {chars_per_token:.2f}")
        print(f"token: word -> 1: {words_per_token:.2f}")

    compress_info = {
        "model_name": model_name,
        "token: character": chars_per_token,
        "vocab_size": vocab_size
    }
    if language == "en":
        compress_info.update({"token: word": words_per_token})
    return compress_info


def moby_dick_books():
    print("Request the book...")
    url = "https://www.gutenberg.org/files/2701/2701-0.txt"
    response = requests.get(url)
    text = response.text
    return text


def download_arxiv_paper(arxiv_id, save_path: str = "./"):
    """Download a paper from arXiv given its ID and save it locally as a PDF."""
    import arxiv
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    pdf_path = os.path.join(save_path, f"{paper.entry_id.split('/')[-1]}.pdf")
    paper.download_pdf(filename=pdf_path)
    print(f"Downloaded: {pdf_path}")
    return pdf_path


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def arxiv_papers(arxiv_id):
    pdf_path = download_arxiv_paper(arxiv_id)
    return extract_text_from_pdf(pdf_path)


def is_cn_text(text: str) -> bool:
    """Check if the text is Chinese."""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def display_in_table(info_list: List[Dict], fmt: str = "github"):
    if not info_list:
        return

    headers = info_list[0].keys()
    rows = [item.values() for item in info_list]
    print(tabulate(rows, headers=headers, tablefmt=fmt))


def compare_compress_rate(
    text: str,
    compare_list=(
        "gpt-4",
        "sp/llama/llama_tokenizer.model",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "moonshot-v1-128k",
        "Qwen/Qwen2-72B",
        "deepseek-ai/DeepSeek-V2",
        "baichuan-inc/Baichuan2-13B-Chat",
        "internlm/internlm2-20b",
    ),
):
    lang = "zh" if is_cn_text(text) else "en"
    info_list = []
    for model_name in compare_list:
        print(f"{model_name} compress rate")
        try:
            info = compress_rate(text, model_name, language=lang, verbose=False)
            info_list.append(info)
        except Exception as e:
            print(f"Skip {model_name} due to error: {e}")
    display_in_table(info_list)


def main():
    book_path = os.path.expanduser("~/Downloads")
    text_dict = {
        "moby_dick_books": moby_dick_books(),
        "PaLM paper": arxiv_papers("2204.02311"),
        "CSAPP": extract_text_from_pdf(os.path.join(book_path, "csapp.pdf")),
        "TB1": extract_text_from_pdf(os.path.join(book_path, "three_body_1.pdf")),
        "TB2": extract_text_from_pdf(os.path.join(book_path, "three_body_2.pdf")),
        "TB3": extract_text_from_pdf(os.path.join(book_path, "three_body_3.pdf")),
    }
    for k, text in text_dict.items():
        print(f"Compress rate of {k}")
        compare_compress_rate(text)
        print("--" * 30)


if __name__ == "__main__":
    main()
