from typing import Dict, List, Union, Any

MODEL_TO_ID = {
    "bert_base": "bert-base-uncased",
    "bert_large": "bert-large-uncased",
    "roberta_base": "roberta-base",
    "roberta_large": "roberta-large",
    "albert_base": "albert-base-v1",
    "albert_large": "albert-large-v1",
}

POSTGRES_TYPE_TO_PYTHON = {
    'text': str,
    'ARRAY': list,
    'INTEGER': int,
    'double precision': float
}

DB_HOST = "127.0.0.1"
DB_PORT = "5432"
# fill in your DB password
DB_PASSWORD = "" # TODO remove from commit.

# ppAdmin port - 65086
