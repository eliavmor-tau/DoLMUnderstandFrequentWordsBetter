from config.config import *
from data.postgres_object import PostgresClient
import json
import os
from typing import Dict, List, Union, Any
import pandas as pd


class WordNetDataReader:
    def __init__(self, json_path):
        if not os.path.isfile(json_path):
            raise(f"Can't create DataReader for {json_path}. File does not exists.")
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.nodes_by_depth = {}
        for node, node_data in self.data.items():
            if node_data["depth"] not in self.nodes_by_depth:
                self.nodes_by_depth[node_data["depth"]] = list()
            self.nodes_by_depth[node_data["depth"]].append(node)

    def get_root(self):
        return self.nodes_by_depth[0][0]

    def get_nodes_by_depth(self, depth: int) -> List[str]:
        if depth in self.nodes_by_depth:
            return self.nodes_by_depth[depth]
        return list()

    def generate_sentences(self, base_sent, mask="<MASK>", depth=None):
        if depth is None:
            return self.data.keys()
        words = self.nodes_by_depth[depth] if depth in self.nodes_by_depth else []
        sentences = []
        for word in words:
            sentences.append(base_sent.replace(mask, word))
        return sentences


class DataReader:
    def __init__(self, host, port, password):
        self.postgres_client = PostgresClient(host=host, port=port, password=password)

    def _get_info_on_head(self, head, table, condition_eq=True, limit="", order_by=""):
        assert table.startswith("ConceptNet")
        df = pd.DataFrame(columns=["head", "tail", "weight"])
        data = self.postgres_client.select(table=table, select='tail, weight',
                                           condition=f"head{'=' if condition_eq else '!='}'{head}'", order_by=order_by,
                                           limit=limit)
        for tail, weight in data:
            df = df.append({"head": head, "tail": tail, "weight": weight}, ignore_index=True)
        return df

    def _get_info_on_tail(self, tail, table, condition_eq=True, limit="", order_by=""):
        assert table.startswith("ConceptNet")
        df = pd.DataFrame(columns=["name"])
        data = self.postgres_client.select(table=table, select='head, weight',
                                           condition=f"tail{'=' if condition_eq else '!='}'{tail}'", order_by=order_by,
                                           limit=limit)
        for head, weight in data:
            df = df.append({"name": head}, ignore_index=True)
        return df

    def _get_hyponyms_from_wordnet(self, name, limit="", order_by=""):
        data = self.postgres_client.select(table="WordNet", select='hyponyms',
                                           condition=f"name='" + name + "'", limit=limit, order_by=order_by)
        if len(data):
            data = data[0][0]
        hyponyms = set(data)
        for hyponym in data:
            hyponyms = hyponyms.union(self._get_hyponyms_from_wordnet(hyponym))
        return hyponyms

    def generate_is_a_sentences(self, category, base_sent, entity_mask, category_mask, limit=""):
        # data = self._get_info_on_tail(tail=category, table="ConceptNetIsA", condition_eq=True, limit=limit, order_by="weight")
        data = pd.DataFrame()
        hyponyms = self._get_hyponyms_from_wordnet(name=category)
        for hyponym in hyponyms:
            hyponym = hyponym.replace("-", "_")
            data = data.append({"name": hyponym}, ignore_index=True)
        sentences = []
        for idx, row in data.iterrows():
            sentences.append(base_sent.replace(entity_mask, row["name"]).replace(category_mask, category))

        data.insert(loc=0, column="sentence", value=sentences)
        return data


if __name__ == "__main__":
    data_reader = DataReader(host=DB_HOST, port=DB_PORT, password=DB_PASSWORD)
