from config.config import *
from nltk.corpus import wordnet as wn
from graphviz import Digraph
import json
import os
from data.postgres_object import PostgresClient


class SynsetTree:

    def __init__(self, synset: Union[str, object]):
        if isinstance(synset, str):
            self.synset = wn.synsets(synset)[0]
        else:
            self.synset = synset

        self.name = self.synset.name().partition('.')[0]
        self.pos = self.synset.pos()
        self.hyponyms = list()
        self.hypernyms = list()
        self.wn_hyponyms = self.synset.hyponyms()

        for hyponym in self.wn_hyponyms:
            self.hyponyms.append(SynsetTree(hyponym))

    def get_nodes_count(self) -> int:
        return 1 + sum([hyponym.get_nodes_count() for hyponym in self.hyponyms])

    def _build_graph(self, graph: object) -> object:
        for hyponym in self.hyponyms:
            graph.node(hyponym.name, hyponym.name)
            graph.edge(tail_name=self.name, head_name=hyponym.name)
            graph = hyponym._build_graph(graph)
        return graph

    def plot_tree(self, output_path: str = "") -> None:
        graph = Digraph()
        graph.node(self.name, self.name)
        for hyponym in self.hyponyms:
            graph.node(hyponym.name, hyponym.name)
            graph.edge(tail_name=self.name, head_name=hyponym.name)
            graph = hyponym._build_graph(graph)

        if not output_path:
            output_path = os.path.join("synset_trees", self.name)
        graph.render(output_path)

    def _build_json(self, tree_info: Dict[str, Any], depth: int, hypernyms: List[object]) -> None:

        if self.name not in tree_info:
            tree_info[self.name] = dict()
            tree_info[self.name]["hypernyms"] = hypernyms
            tree_info[self.name]["depth"] = depth
            tree_info[self.name]["hyponyms"] = [hyponym.name for hyponym in self.hyponyms]
            for hyponym in self.hyponyms:
                if hyponym.name in tree_info:
                    tree_info[hyponym.name]["hypernyms"].append(self.name)
                else:
                    hyponym._build_json(tree_info=tree_info, depth=depth+1, hypernyms=[self.name])

    def to_json(self):
        hypernyms = []
        depth = 0
        name = self.name
        hyponyms = [hyponym.name for hyponym in self.hyponyms]
        info = {name: {"depth": depth, "hypernyms": hypernyms, "hyponyms": hyponyms}}
        for hyponym in self.hyponyms:
            hyponym_parents = [name]
            hyponym._build_json(tree_info=info, hypernyms=hyponym_parents, depth=depth+1)
        return info

    def upload_tree_to_db(self, postgres_client: PostgresClient, allow_update=False) -> None:
        tree_json = self.to_json()
        for k, v in tree_json.items():
            insertion_values = {"name": k, "hypernyms": v["hypernyms"], "hyponyms": v["hyponyms"]}
            postgres_client.insert(table="WordNet", insertion_value=insertion_values, scheme="public",
                                   allow_update=allow_update)


class WordNetObj:

    @staticmethod
    def get_entity_hypernyms(entity):
        synset = wn.synsets(entity)
        hypernyms = set()
        hyper = lambda x: x.hypernyms()

        if synset:
            for s in synset:
                s_name = s.name().partition('.')[0]
                if s_name == entity:
                    hypernyms = hypernyms.union({x.name().partition('.')[0] for x in s.closure(hyper)})
        return hypernyms

    @staticmethod
    def get_entity_hyponyms(entity, similarity_threshold=0.14):
        synset = wn.synsets(entity)
        hyponyms = set()
        hypo = lambda x: x.hyponyms()

        if synset:
            for s in synset:
                s_name = s.name().partition('.')[0]
                if s_name != entity:
                    print("sysnet name != entity", s_name, entity)
                else:
                    hyponyms = hyponyms.union({x.name().partition('.')[0] for x in s.closure(hypo) if s.path_similarity(x, simulate_root=False) < similarity_threshold})
        return hyponyms


if __name__ == "__main__":
    # synset_tree = SynsetTree("first")
    # print(synset_tree.to_json())
    entity = "fish"
    hyponyms = WordNetObj.get_entity_hyponyms(entity, similarity_threshold=0.4)
    hypernyms = WordNetObj.get_entity_hypernyms(entity)
    for x in sorted(hyponyms):
        if "-" not in x and "_" not in x:
            print(x)
    print("*" * 100)
    for x in sorted(hypernyms):
        print(x)
    # with open("animal_skin.json", "w") as f:
    #     json.dump(synset_tree.to_json(), f)
    # postgres_client = PostgresClient(password=DB_PASSWORD)
    # synset_tree.upload_tree_to_db(postgres_client)
    # postgres_client.close()
