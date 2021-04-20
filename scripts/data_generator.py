from scripts.tokenizer import Tokenizer
from models.mlm_models import *
from transformers import AutoConfig, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.nn.functional import softmax
from scripts.external_data_reader import DataReader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
from nltk.corpus import wordnet as wn
from scripts.wordnet_parser import WordNetObj
from scripts.concept_net import ConceptNetObj
from itertools import permutations, combinations
from collections import defaultdict
import time
import os

# model_name = 'roberta-large'
# mc_mlm = True
# config = AutoConfig.from_pretrained(model_name)
# tokenizer = Tokenizer(model_name)
# model = TransformerMaskedLanguageModel(vocab=config, model_name=model_name, multi_choice=mc_mlm)
# data_reader = DataReader(host=DB_HOST, port=DB_PORT, password=DB_PASSWORD)
# concept_net = ConceptNetObj()
# wordnet = WordNetObj()
#
#
# def test_sentence_mc_mlm(sentences, target_index, multi_choice_answers, k=1, max_length=128):
#     input_ids = []
#     attention_mask = []
#     target_index = torch.Tensor(target_index).view(-1, 1).long()
#     for sent in sentences:
#         tokenized = tokenizer.encode(sent, add_special_tokens=True, max_length=max_length)
#         input_ids.append(tokenized.input_ids)
#         attention_mask.append(tokenized.attention_mask)
#     input_ids = torch.vstack(input_ids)
#     attention_mask = torch.vstack(attention_mask)
#     batch_size = input_ids.size()[0]
#
#     indices = torch.arange(input_ids.size()[1]).repeat(batch_size).view((batch_size, max_length))
#     mask_loc = input_ids == tokenizer.mask_token_id()
#     mask_loc = indices[mask_loc].view(-1)
#
#     # MASK for MC-MLM
#     multi_choice_label_ids = tokenizer.convert_tokens_to_ids(multi_choice_answers).long()
#     all_indices_mask = torch.zeros((batch_size, max_length, config.vocab_size))
#     all_indices_mask[:, :, multi_choice_label_ids] = 1
#     multi_choice_label_ids = multi_choice_label_ids.repeat(batch_size).view((batch_size, -1))
#
#     labels = torch.tensor([-100] * max_length * batch_size).view((batch_size, max_length))
#     target_index = target_index.view(-1)
#
#     for i in range(batch_size):
#         labels[i, mask_loc[i]] = multi_choice_label_ids[i, target_index[i]]
#
#     output = model(input_ids=input_ids, token_type_ids=None, all_indices_mask=all_indices_mask, labels=labels, attention_mask=attention_mask)
#     output_softmax = softmax(output["logits"], dim=2)
#     print(output_softmax.shape)
#     argmax_output = torch.topk(output_softmax, k=k, dim=2)[1]
#     predictions = []
#     for i in range(batch_size):
#         mask_index = mask_loc[i]
#         masked_predictions = argmax_output[0][mask_index]
#         for pred in masked_predictions:
#             predictions.append((tokenizer.convert_ids_to_tokens(pred.item()).replace('Ġ', ''), output_softmax[0][mask_index][pred.item()].item()))
#     return predictions
#
#
# def test_sentence_mlm(sentence, mask_index, batch_size, k=1):
#     input_sent = tokenizer.encode(sentence, add_special_tokens=True)  # TODO: move to config?
#     mask_index += len(input_sent) - 2 - len(sentence.split(" "))
#     if sentence.endswith("."):
#         mask_index -= 1
#     input_sent[mask_index] = tokenizer.mask_token_id()
#     input_tensor = torch.tensor(input_sent).view((batch_size, -1))
#     token_type_ids = torch.tensor([0] * len(input_sent)).view((batch_size, -1))
#
#     output = model(input_ids=input_tensor, token_type_ids=token_type_ids)
#
#     output_softmax = softmax(output["logits"], dim=2)
#     argmax_output = torch.topk(output_softmax, k=k, dim=2)[1]
#     masked_predictions = argmax_output[0][mask_index]
#     predictions = []
#     for pred in masked_predictions:
#         predictions.append((tokenizer.convert_ids_to_tokens(pred.item()).replace('Ġ', ''),
#                             output_softmax[0][mask_index][pred.item()].item()))
#     return predictions


def filter_word_in_model_vocab(tokenizer, words):
    known_words = [w for w in words if tokenizer.convert_tokens_to_ids(w) != tokenizer.unk_id()]
    return known_words


def filter_word_not_in_model_vocab(tokenizer, words):
    known_words = [w for w in words if tokenizer.convert_tokens_to_ids(w) == tokenizer.unk_id()]
    return known_words


def generate_questions_from_csv(csv_path):
    print(csv_path)
    df = pd.read_csv(csv_path)
    data = {"question": [], "label": []}
    if "entity" not in df.columns:
        return {"question": list(df.question.values),
                "label": list(df.label.values)}
    for row in df.iterrows():
        row = row[1]
        entity = row["entity"]
        for question, label in row.items():
            if label == entity:
                continue
            question = question.replace("<entity>", entity).lower()
            data["question"].append(question)
            data["label"].append("Yes" if label > 0 else "No")
    return data


def merge_questions(csv_paths, output_path="csv/train_questions.csv", split=True, p=0.7):
    data = {"question": [], "label": []}
    for csv_path in csv_paths:
        csv_data = generate_questions_from_csv(csv_path=csv_path)
        data["question"] += csv_data["question"]
        data["label"] += csv_data["label"]

    questions = data["question"]
    labels = data["label"]
    all_questions = pd.DataFrame.from_dict({"question": questions, "label": labels})
    yes_questions = all_questions[all_questions.label == "Yes"]
    no_questions = all_questions[all_questions.label == "No"]
    N = min(len(yes_questions), len(no_questions))
    print("yes count", len(yes_questions))
    print("no count", len(no_questions))
    print("N", N)
    yes_questions_df = yes_questions.sample(n=N)
    no_questions_df = no_questions.sample(n=N)
    yes_questions, yes_label = yes_questions_df.question.values, yes_questions_df.label.values
    no_questions, no_label = no_questions_df.question.values, no_questions_df.label.values
    if split:
        train_size = int(N * p)
        train_indices = np.random.choice(a=np.arange(N), size=train_size, replace=False)
        val_indices = np.array(list(set(np.arange(N)).difference(set(train_indices))))
        train_df = pd.DataFrame.from_dict(
            {"question": np.hstack([yes_questions[train_indices], no_questions[train_indices]]),
             "label": np.hstack([yes_label[train_indices], no_label[train_indices]])})
        val_df = pd.DataFrame.from_dict(
            {"question": np.hstack([yes_questions[val_indices], no_questions[val_indices]]),
             "label": np.hstack([yes_label[val_indices], no_label[val_indices]])})
        train_df.to_csv("../csv/train_questions.csv")
        val_df.to_csv("../csv/val_questions.csv")
    else:
        indices = np.random.choice(a=np.arange(len(yes_questions) + len(no_questions)), size=2*N, replace=False)
        print("Number of samples: ", len(set(indices)))
        questions = np.hstack([yes_questions, no_questions])[indices]
        print("len(questions)", len(questions))
        labels = np.hstack([yes_label, no_label])[indices]
        final_df = pd.DataFrame.from_dict(
            {"question": questions,
             "label": labels})
        final_df.to_csv(output_path)
    return data


def split_data(csv_path, prefix="", p=0.8):
    all_questions = pd.read_csv(csv_path)
    yes_questions = all_questions[all_questions.label == "Yes"]
    no_questions = all_questions[all_questions.label == "No"]
    N = min(len(yes_questions), len(no_questions))
    yes_questions_df = yes_questions.sample(n=N)
    no_questions_df = no_questions.sample(n=N)
    yes_questions, yes_label = yes_questions_df.question.values, yes_questions_df.label.values
    no_questions, no_label = no_questions_df.question.values, no_questions_df.label.values
    train_size = int(N * p)
    train_indices = np.random.choice(a=np.arange(N), size=train_size, replace=False)
    val_indices = np.array(list(set(np.arange(N)).difference(set(train_indices))))
    train_df = pd.DataFrame.from_dict(
        {"question": np.hstack([yes_questions[train_indices], no_questions[train_indices]]),
         "label": np.hstack([yes_label[train_indices], no_label[train_indices]])})
    val_df = pd.DataFrame.from_dict(
        {"question": np.hstack([yes_questions[val_indices], no_questions[val_indices]]),
         "label": np.hstack([yes_label[val_indices], no_label[val_indices]])})
    train_df.to_csv(f"../csv/{prefix}train_questions.csv")
    val_df.to_csv(f"../csv/{prefix}val_questions.csv")
    print("Number of questions", N)


def animal_accuracy(animal, result_df):
    animal_df = result_df[[f" {animal} " in question or f" {animal}'s" in question for question in result_df.question]]
    accuracy = len(animal_df[animal_df.model_answer == animal_df.true_answer]) / len(animal_df)
    yes_count = len(animal_df[animal_df.model_answer == "Yes"])
    no_count = len(animal_df[animal_df.model_answer == "No"])
    return animal, "{:.2f}".format(accuracy), yes_count, no_count


def clean_question(question):
    question = question.replace("?", "")
    question = question.replace(".", "")
    question = question.replace(",", "")
    question = question.replace(":", "")
    question = question.replace(";", "")
    question = question.replace("'", "")
    question = question.replace('"', "")
    question = question.replace('=', "")
    question = question.replace('-', "")
    question = question.replace(">", "")
    question = question.replace(")", "")
    question = question.replace("(", "")
    question = question.replace("/", "")
    question = question.replace("\\", "")
    question = question.replace("@", "")
    question = question.replace("#", "")
    question = question.replace("%", "")
    question = question.replace("&", "")
    question = question.replace("*", "")
    return question.lower()


def filter_questions():
    properties = {"animal", 'scale', 'scales', 'fur', 'hair', 'hairs', 'tail', 'legs', 'leg', 'fly',
                  'flies', "flying", 'climb', 'climbs', 'carnivore', 'herbivore', 'omnivore', 'bones', 'bone', 'beak',
                  'teeth', 'feathers', 'feather', 'horn', 'horns', 'hooves', 'claws', 'blooded', "wing", "wings"}

    files = ["animals_have_a_beak", "animals_have_horns", "animals_have_fins", "animals_have_scales",
             "animals_have_wings", "animals_have_feathers", "animals_have_fur",
             "animals_have_hair", "animals_live_underwater", "animals_can_fly",
             "animals_dont_have_a_beak", "animals_dont_have_horns", "animals_dont_have_fins",
             "animals_dont_have_scales",
             "animals_dont_have_wings", "animals_dont_have_feathers", "animals_dont_have_fur",
             "animals_dont_have_hair", "animals_dont_live_underwater", "animals_cant_fly", "animals", "sanity"]
    animals = set()
    for file in files:
        animals = animals.union(set(pd.read_csv(f"csv/{file}.csv").entity.values))
    animals = animals.union({animal + "s" for animal in animals})
    with open('../json/twenty_questions_it_replace_rand_split_train.jsonl', 'r') as f:
        lines = f.readlines()
        data = {"question": [], "label": []}
        for line in lines:
            line_dict = json.loads(line)
            question, answer = line_dict["phrase"], line_dict["answer"]
            question = clean_question(question)
            split_question = set(question.split(' '))

            if not(len(animals.intersection(split_question)) or len(properties.intersection(split_question))):
                if question not in data["question"]:
                    data["question"].append(question)
                    data["label"].append("Yes" if answer else "No")
            else:
                print(question, answer)

    with open('../json/conceptnet_train.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_dict = json.loads(line)
            question, answer = line_dict["phrase"], line_dict["answer"]
            question = clean_question(question)
            split_question = set(question.split(' '))

            if not (len(animals.intersection(split_question)) or len(properties.intersection(split_question))):
                if question not in data["question"]:
                    data["question"].append(question)
                    data["label"].append("Yes" if answer else "No")
            else:
                print(question, answer)
    extra_questions = pd.read_csv("csv/manual_questions.csv")
    for row in extra_questions.iterrows():
        row = row[1]
        question, answer = row["question"], row["label"]
        question = clean_question(question)
        split_question = set(question.split(' '))

        if not (len(animals.intersection(split_question)) or len(properties.intersection(split_question))):
            if question not in data["question"]:
                data["question"].append(question)
                data["label"].append(answer)
            else:
                print("Question is already exists:", question)
        else:
            print("Intersection failed", question)
            print(animals.intersection(split_question))
            print(properties.intersection(split_question))


    questions_df = pd.DataFrame.from_dict(data)
    yes_num_questions = len(questions_df[questions_df["label"] == "Yes"])
    no_num_questions = len(questions_df[questions_df["label"] == "No"])
    print(yes_num_questions, no_num_questions)
    N = min(yes_num_questions, no_num_questions)
    print(N)
    new_yes_questions = questions_df[questions_df["label"] == "Yes"].sample(n=N, replace=False)
    new_no_questions = questions_df[questions_df["label"] == "No"].sample(n=N, replace=False)
    questions_df = pd.concat([new_yes_questions, new_no_questions], axis=0, ignore_index=True)
    questions_df.to_csv("../csv/merged_train_questions_no_animals_and_fruits.csv")


def aggregate_results_by_animal(result_df, animals_df):
    animals = animals_df["entity"].values
    results_by_animal = {"animal": [], "accuracy": [], "yes_count": [], "no_count": []}
    for animal in animals:
        animal, accuracy, yes_count, no_count = animal_accuracy(animal, result_df)
        results_by_animal["animal"].append(animal)
        results_by_animal["accuracy"].append(accuracy)
        results_by_animal["yes_count"].append(yes_count)
        results_by_animal["no_count"].append(no_count)
    return pd.DataFrame.from_dict(results_by_animal)


def aggregate_results_by_question(result_df, animals_df):
    questions = [q.lower() for q in list(animals_df.columns.values)]
    questions.remove("entity")
    animals = set(animals_df["entity"].values)
    results_by_question = {q: {"accuracy": 0, "yes_count": 0, "no_count": 0} for q in questions}
    for idx, row in result_df.iterrows():
        question = row["question"]
        status = row["model_answer"] == row["true_answer"]
        is_yes = row["model_answer"] == "Yes"
        for animal in animals:
            if "'s" in question:
                question = question.replace(f" {animal}'s", " <entity>'s")
            else:
                orig_questions = question
                question = question.replace(f" {animal} ", " <entity> ")
                if question == orig_questions:
                    question = question.replace(f"{animal}", "<entity>")

        for q in results_by_question.keys():
            if question == q:
                results_by_question[q]["accuracy"] += int(status)
                results_by_question[q]["yes_count"] += int(is_yes)
                results_by_question[q]["no_count"] += abs(int(is_yes) - 1)
                break

    for q in results_by_question.keys():
        results_by_question[q]["accuracy"] /= float(results_by_question[q]["yes_count"] + results_by_question[q]["no_count"])
        print(q, results_by_question[q])
    return pd.DataFrame.from_dict(results_by_question)


def summarize_results(animals_csv_path, results_csv_path):
    animals_df = pd.read_csv(animals_csv_path)
    result_df = pd.read_csv(results_csv_path)
    results_by_animal = aggregate_results_by_animal(result_df, animals_df)
    results_by_question = aggregate_results_by_question(result_df, animals_df)
    results_by_animal.sort_values(axis=0, by=["accuracy"])
    results_by_animal.to_csv(results_csv_path.replace(".csv", "_by_animal.csv"))
    results_by_question.to_csv(results_csv_path.replace(".csv", "_by_question.csv"))


def plot_df(csv_path, output_path=""):
    df = pd.read_csv(csv_path)
    df.sort_values(axis=0, by=['accuracy'])
    rows, cols = [row[1] for row in df.iterrows()], list(df.columns.values)
    if "animal" in cols:
        cols.remove("animal")
    if "Unnamed: 0" in cols:
        cols.remove('Unnamed: 0')

    figure, ax = plt.subplots(figsize=(10, 20))
    cell_text = []
    for row in rows:
        cell_text.append(["{:.3f}".format(row[c]) for c in cols])

    # Add a table at the bottom of the axes
    plt.table(cellText=cell_text, rowLabels=[row["animal"] for row in rows], colLabels=cols, loc="center")
    figure.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    figure.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def run_summarize_results():
    files = ["animals_have_a_beak", "animals_have_horns", "animals_have_fins",
             "animals_have_wings", "animals_have_feathers", "animals_have_fur",
             "animals_have_hair", "animals_live_underwater", "animals_can_fly",
             "animals_dont_have_a_beak", "animals_dont_have_horns", "animals_dont_have_fins",
             "animals_dont_have_wings", "animals_dont_have_feathers", "animals_dont_have_fur",
             "animals_dont_have_hair", "animals_dont_live_underwater", "animals_cant_fly", "sanity"]
    files = ["animals_even_properties"]
    for file in files:
        print(f"summarize {file}")
        summarize_results(animals_csv_path=f"../csv/{file}.csv",
                          results_csv_path=f"../csv/results/{file}_questions_result.csv")


def run_generate_questions():
    files = ["animals_have_a_beak", "animals_have_horns", "animals_have_fins",
             "animals_have_wings", "animals_have_feathers", "animals_have_fur",
             "animals_have_hair", "animals_live_underwater", "animals_can_fly",
             "animals_dont_have_a_beak", "animals_dont_have_horns", "animals_dont_have_fins",
             "animals_dont_have_wings", "animals_dont_have_feathers", "animals_dont_have_fur",
             "animals_dont_have_hair", "animals_dont_live_underwater", "animals_cant_fly", "old/food", "old/furniture",
             "old/musical_instruments", "old/vehicle", "sanity"]
    files = ["animals_even_properties"]
    for file in files:
        questions = generate_questions_from_csv(csv_path=f"csv/{file}.csv")
        questions = pd.DataFrame.from_dict(questions)
        questions.to_csv(f"../csv/{file}_questions.csv")


if __name__ == "__main__":
    run_summarize_results()