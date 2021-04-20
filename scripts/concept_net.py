import requests
from scripts.postgres_object import PostgresClient
from scripts import utilities
from config.config import *


class ConceptNetObj:
    def __init__(self):
        self.postgres_client = PostgresClient(password=DB_PASSWORD)

    def _update_postgress_with_all_wordnet(self):
        entity_info = ["PartOf", "IsA", "HasA", "CapableOf", "AtLocation", "HasProperty", "MadeOf"]
        limit = 10000
        rows = self.postgres_client.select(table="WordNet", select="name")
        for idx, row in enumerate(rows[23221:]):
            subject = row[0]
            obj = requests.get(f'http://api.conceptnet.io/c/en/{subject}?limit={limit}')
            if obj is None:
                continue
            else:
                obj = obj.json()

            while "view" in obj and 'nextPage' in obj["view"]:
                for edge in obj["edges"]:
                    relation = edge['rel']['label']
                    if relation in entity_info:
                        start_split, end_split = edge['start']['@id'].split("/"), edge['end']['@id'].split("/")
                        A = str(start_split[-1] if len(start_split[-1]) > 1 else start_split[-2])
                        B = str(end_split[-1] if len(end_split[-1]) > 1 else end_split[-2])

                        md5 = utilities.string_to_md5(A+B)
                        weight = edge["weight"]
                        insertion_value = {"head": A, "tail": B, "weight": weight, "md5": md5}

                        self.postgres_client.insert(table=f"ConceptNet{relation}", insertion_value=insertion_value)
                obj = requests.get(f"http://api.conceptnet.io{obj['view']['nextPage']}").json()

            for edge in obj["edges"]:
                relation = edge['rel']['label']
                if relation in entity_info:
                    start_split, end_split = edge['start']['@id'].split("/"), edge['end']['@id'].split("/")
                    A = str(start_split[-1] if len(start_split[-1]) > 1 else start_split[-2])
                    B = str(end_split[-1] if len(end_split[-1]) > 1 else end_split[-2])

                    md5 = utilities.string_to_md5(A + B)
                    weight = edge["weight"]
                    insertion_value = {"head": A, "tail": B, "weight": weight, "md5": md5}
                    self.postgres_client.insert(table=f"ConceptNet{relation}", insertion_value=insertion_value)

    def get_information_on_entity(self, entity, update_db=True):
        limit = 10000
        entity = entity.replace("'", '')
        entity_info = self._get_cached_info_on_entity(entity)
        is_cached = sum([len(v) for k, v in entity_info.items()]) > 0 or \
                    len(self.postgres_client.select(table="ConceptNetWithoutData", condition=f"name='{entity}'")) > 0
        if not is_cached:
            print(entity, is_cached)
        is_added = False
        if not is_cached:
            print("get scripts from conceptnet", entity)
            obj = requests.get(f'http://api.conceptnet.io/c/en/{entity}?limit={limit}')
            if obj is None:
                return
            else:
                obj = obj.json()

            while "view" in obj and 'nextPage' in obj["view"]:
                for edge in obj["edges"]:
                    relation = edge['rel']['label']
                    if relation == "RelatedTo":
                        relation = "IsA"
                    if relation in entity_info:
                        is_added = True
                        start_split, end_split = edge['start']['@id'].split("/"), edge['end']['@id'].split("/")
                        A = str(start_split[-1] if len(start_split[-1]) > 1 else start_split[-2])
                        B = str(end_split[-1] if len(end_split[-1]) > 1 else end_split[-2])

                        md5 = utilities.string_to_md5(A+B)
                        weight = edge["weight"]
                        insertion_value = {"head": A, "tail": B, "weight": weight, "md5": md5}
                        entity_info[relation].append(insertion_value)
                        if update_db:
                            self.postgres_client.insert(table=f"ConceptNet{relation}", insertion_value=insertion_value, allow_update=True)
                obj = requests.get(f"http://api.conceptnet.io{obj['view']['nextPage']}").json()

            for edge in obj["edges"]:
                relation = edge['rel']['label']
                if relation == "RelatedTo":
                    relation = "IsA"

                if relation in entity_info:
                    start_split, end_split = edge['start']['@id'].split("/"), edge['end']['@id'].split("/")
                    A = str(start_split[-1] if len(start_split[-1]) > 1 else start_split[-2])
                    B = str(end_split[-1] if len(end_split[-1]) > 1 else end_split[-2])

                    md5 = utilities.string_to_md5(A + B)
                    weight = edge["weight"]
                    insertion_value = {"head": A, "tail": B, "weight": weight, "md5": md5}
                    entity_info[relation].append(insertion_value)
                    if update_db:
                        self.postgres_client.insert(table=f"ConceptNet{relation}", insertion_value=insertion_value, allow_update=True)

        if not (is_added or is_cached):
            print(f"log {entity} without DATA")
            self.postgres_client.insert(table=f"ConceptNetWithoutData", insertion_value={"name": entity},
                                        allow_update=True)

        return entity_info

    def _get_cached_info_on_entity(self, entity):
        entity_info = {"PartOf": [], "IsA": [], "HasA": [], "CapableOf": [], "AtLocation": [], "HasProperty": [], "MadeOf": []}
        for relation in entity_info:
            data = self.postgres_client.select(table=f"ConceptNet{relation}", select="head, tail, weight",
                                                      condition=f"head='{entity}'")

            for row in data:
                entity_info[relation].append({'head': row[0], 'tail': row[1], 'weight': row[2]})

        return entity_info

if __name__ == "__main__":
    conceptnet = ConceptNetObj()
    entity_info = conceptnet.get_information_on_entity("boa")
    print(sum([len(v) for k, v in entity_info.items()]))
    for relation in entity_info:
        print(relation)
        for data in entity_info[relation]:
            print(data)
        print("*" * 20)