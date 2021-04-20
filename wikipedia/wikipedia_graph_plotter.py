import re
import xml.etree.ElementTree as ET
from collections import Counter
import os
import threading
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare

def find_word_in_text(word, text):
    indices = set()
    indices.add(text.find(f" {word} "))
    indices.add(text.find(f" {word}."))
    indices.add(text.find(f" {word}s"))
    indices.add(text.find(f" {word}:"))
    indices.add(text.find(f'"{word}"'))
    indices.add(text.find(f' {word}?'))
    indices.add(text.find(f' {word}-'))
    indices.add(text.find(f' {word},'))
    indices.add(text.find(f' {word};'))
    indices.remove(-1)
    if len(indices) == 0:
        return -1
    else:
        return min(indices)


def plot_occurrence_by_property(exact=False):
    df_pairs = [
        ("../csv/results/animals_cant_fly_questions_result_by_animal.csv", "../csv/results/animals_can_fly_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_a_beak_questions_result_by_animal.csv", "../csv/results/animals_have_a_beak_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_feathers_questions_result_by_animal.csv", "../csv/results/animals_have_feathers_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_fins_questions_result_by_animal.csv", "../csv/results/animals_have_fins_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_fur_questions_result_by_animal.csv", "../csv/results/animals_have_fur_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_hair_questions_result_by_animal.csv", "../csv/results/animals_have_hair_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_horns_questions_result_by_animal.csv", "../csv/results/animals_have_horns_questions_result_by_animal.csv"),
        # ("../csv/results/animals_dont_have_scales_questions_result_by_animal.csv", "../csv/results/animals_have_scales_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_wings_questions_result_by_animal.csv", "../csv/results/animals_have_wings_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_live_underwater_questions_result_by_animal.csv", "../csv/results/animals_live_underwater_questions_result_by_animal.csv"),
    ]
    title = [
        "Can animal fly?",
        "Does a animal have a beak?",
        "Does a animal have feathers?",
        "Does a animal have fins?",
        "Does a animal have fur?",
        "Does a animal have hair?",
        "Does a animal have horns?",
        # "Does a animal have scales?",
        "Does a animal have wings?",
        "Does a animal live underwater?",
    ]

    property = [
        "fly",
        "beak",
        "feathers",
        "fins",
        "fur",
        "hair",
        "horns",
        # "scales",
        "wings",
        "underwater",
    ]

    with open("wiki_unigram_dont_delete.pkl", "rb") as f:
        wiki_unigram = pickle.load(f)

    data = []
    for idx, df_pair in enumerate(df_pairs):
        animal_group_A = pd.read_csv(df_pair[0])
        animal_group_B = pd.read_csv(df_pair[1])
        if exact:
            total_animals = len(animal_group_A[animal_group_A["accuracy"].values == 1]) + \
                            len(animal_group_B[animal_group_B["accuracy"].values == 1]) + \
                            len(animal_group_A[animal_group_A["accuracy"].values == 0]) +\
                            len(animal_group_B[animal_group_B["accuracy"].values == 0])
        else:
            total_animals = len(animal_group_A.animal.values) + len(animal_group_B.values)
        current_property = property[idx]
        if exact:
            accuracy = ((animal_group_A["accuracy"].values == 1).sum() + (animal_group_B["accuracy"].values == 1).sum()) / total_animals
        else:
            accuracy = ((animal_group_A["accuracy"].values > 0.5).sum() + (animal_group_B["accuracy"] > 0.5).sum()) / total_animals
        print(property[idx], accuracy, f"{total_animals} / {len(animal_group_A) + len(animal_group_B)}")

        property_count = wiki_unigram[current_property] if current_property in wiki_unigram else 0
        data.append((property_count, accuracy, current_property))

    data.sort(key=lambda x: x[0])
    plt.figure(figsize=(15, 5))
    X = np.array([x[0] for x in data])
    Y = np.array([x[1] for x in data])
    A = np.hstack([np.ones((len(X), 1)), X.reshape((-1, 1))])
    b = Y.reshape((-1, 1))
    res = np.linalg.lstsq(a=A, b=b)
    a, b = res[0][0], res[0][1]
    plt.plot(X, X*b + a, "--", color="red", label="linear regression")
    labels = [x[2] for x in data]
    plt.scatter(X, Y)
    for i in range(len(X)):
        if i < len(X) - 1:
            slope = Y[i+1] - Y[i]
            if slope < 0 or labels[i] == 'feathers':
                plt.annotate(labels[i], (X[i] - 500, Y[i] + 0.02))
            else:
                plt.annotate(labels[i], (X[i] - 500, Y[i] - 0.05))
        else:
            plt.annotate(labels[i], (X[i] - 500, Y[i] + 0.02))
    plt.ylim(0, 1)
    plt.xlabel("Property Occurrence Count in Wikipedia")
    plt.ylabel("Accuracy")
    x_ticks = np.arange(0, 80000+1, 80000 // 4)
    x_ticks_labels = [f"{tick // 1000}K" for tick in x_ticks]
    plt.xticks(x_ticks, x_ticks_labels)
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()



def plot_cooccurrence():
    df_pairs = [
        ("../csv/results/animals_cant_fly_questions_result_by_animal.csv", "../csv/results/animals_can_fly_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_a_beak_questions_result_by_animal.csv", "../csv/results/animals_have_a_beak_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_feathers_questions_result_by_animal.csv", "../csv/results/animals_have_feathers_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_fins_questions_result_by_animal.csv", "../csv/results/animals_have_fins_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_fur_questions_result_by_animal.csv", "../csv/results/animals_have_fur_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_hair_questions_result_by_animal.csv", "../csv/results/animals_have_hair_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_horns_questions_result_by_animal.csv", "../csv/results/animals_have_horns_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_wings_questions_result_by_animal.csv", "../csv/results/animals_have_wings_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_live_underwater_questions_result_by_animal.csv", "../csv/results/animals_live_underwater_questions_result_by_animal.csv"),

    ]

    property = [
        "fly",
        "beak",
        "feather",
        "fin",
        "fur",
        "hair",
        "horn",
        "wing",
        "underwater",
    ]

    with open("wiki_word_to_sentences.pkl", "rb") as f:
        wiki_text_chunks_by_animal = pickle.load(f)

    all_yes_count_percentage = []
    all_co_occurrence_count = []
    all_animals = []
    for idx, df_pair in enumerate(df_pairs):
        co_occurrence_count, yes_count_percentage, animals = co_occurrence_helper(df_pair, wiki_text_chunks_by_animal, property[idx])
        all_yes_count_percentage.append(yes_count_percentage)
        all_co_occurrence_count.append(co_occurrence_count)
        all_animals.append(animals)

    all_co_occurrence_count = np.hstack(all_co_occurrence_count)
    all_yes_count_percentage = np.hstack(all_yes_count_percentage)
    data = list(zip(all_co_occurrence_count, all_yes_count_percentage))
    data = sorted(data, key=lambda x: x[0])
    all_co_occurrence_count = np.array([x[0] for x in data])
    all_yes_count_percentage = np.array([x[1] for x in data])
    thresholds = np.unique(all_co_occurrence_count)
    best_threshold = 0
    best_accuracy = 0
    for th in thresholds:
        predictions = (all_co_occurrence_count > th).astype(np.int)
        accuracy = (predictions == all_yes_count_percentage).sum() / len(all_yes_count_percentage)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = th
            print(f"Found better threshold {best_threshold} accuracy {best_accuracy}")
            print("-" * 20)
    print(f"Best Yes/No Classifier {best_threshold} with accuracy {best_accuracy}")
    print(f"Yes count {(all_yes_count_percentage==1).sum()}")
    print(f"No count {(all_yes_count_percentage==0).sum()}")
    data = {"co_occurrence": all_co_occurrence_count, "Model Answer": all_yes_count_percentage}
    data = pd.DataFrame.from_dict(data)
    data["co_occurrence"][data["Model Answer"] == 1] = "Yes"
    data["co_occurrence"][data["Model Answer"] == 0] = "No"
    max_power = int(np.ceil(np.log2(np.max(all_co_occurrence_count))))
    bins = [2 ** i for i in range(max_power + 1)]
    buckets = [0] * len(bins)
    counts = [0] * len(bins)
    for i, p in enumerate(all_co_occurrence_count):
        if p <= 2:
            buckets[0] += all_yes_count_percentage[i]
            counts[0] += 1
        else:
            buckets[int(np.floor(np.log2(p)))] += all_yes_count_percentage[i]
            counts[int(np.floor(np.log2(p)))] += 1
    Y = np.array(buckets) / np.array(counts)
    for i in range(len(bins)):
        print(f"Bin {bins[i-1] if i > 0 else 0} - {bins[i]} | Count {counts[i]} | Yes Percentage {Y[i]}")

    plt.plot((bins + np.hstack([[0], bins[:-1]])) / 2., Y, "o-")
    plt.ylabel("Probability of 'Yes' Answer")
    plt.xlabel("Aggregated (animal, property) pair Co-Occurrence Count")
    plt.grid()
    plt.show()
    plt.close()

    plt.hist(all_co_occurrence_count[all_yes_count_percentage == 1], color="blue", alpha=0.4, bins=256, label="LM answers Yes")
    plt.hist(all_co_occurrence_count[all_yes_count_percentage == 0], color="red", alpha=1, bins=256, label="LM answers No")
    plt.ylim(0, 30)
    plt.xlim(0, 2000)
    plt.ylabel("Frequency")
    plt.xlabel("(animal, property) pair Co-Occurrence Count")
    plt.legend()
    plt.grid()
    plt.show()

def count_property_appears_in_chunks(text_chunks, property):
    count = 0
    for text in text_chunks:
        count += int(find_word_in_text(property, text)) >= 0
    return count



def co_occurrence_helper(df_paths, sentences_by_animal, property):
    df = pd.read_csv(df_paths[0])
    df2 = pd.read_csv(df_paths[1])
    animals = np.hstack([df.animal.values, df2.animal.values])
    accuracy = np.hstack([df["accuracy"].values, df2["accuracy"].values])
    yes_count = np.hstack([df["yes_count"].values, df2["yes_count"].values])
    no_count = np.hstack([df["no_count"].values, df2["no_count"].values])
    co_occurrence_count = np.array([count_property_appears_in_chunks(sentences_by_animal[animal],
                                    property) if animal in sentences_by_animal and len(sentences_by_animal[animal]) \
                                    else 0 for animal in animals])
    mask = np.bitwise_or(accuracy == 1, accuracy == 0)
    p_yes = np.array(yes_count / (yes_count + no_count))
    return co_occurrence_count[mask], p_yes[mask], animals[mask]


def plot_occurrence_by_animal(exact=True):
    df_pairs = [
        ("../csv/results/animals_cant_fly_questions_result_by_animal.csv", "../csv/results/animals_can_fly_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_a_beak_questions_result_by_animal.csv", "../csv/results/animals_have_a_beak_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_feathers_questions_result_by_animal.csv", "../csv/results/animals_have_feathers_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_fins_questions_result_by_animal.csv", "../csv/results/animals_have_fins_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_fur_questions_result_by_animal.csv", "../csv/results/animals_have_fur_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_hair_questions_result_by_animal.csv", "../csv/results/animals_have_hair_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_horns_questions_result_by_animal.csv", "../csv/results/animals_have_horns_questions_result_by_animal.csv"),
        # ("../csv/results/animals_dont_have_scales_questions_result_by_animal.csv", "../csv/results/animals_have_scales_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_have_wings_questions_result_by_animal.csv", "../csv/results/animals_have_wings_questions_result_by_animal.csv"),
        ("../csv/results/animals_dont_live_underwater_questions_result_by_animal.csv", "../csv/results/animals_live_underwater_questions_result_by_animal.csv"),
    ]
    title = [
        "Can animal fly?",
        "Does a animal have a beak?",
        "Does a animal have feathers?",
        "Does a animal have fins?",
        "Does a animal have fur?",
        "Does a animal have hair?",
        "Does a animal have horns?",
        # "Does a animal have scales?",
        "Does a animal have wings?",
        "Does a animal live underwater?",
    ]

    property = [
        "fly",
        "beak",
        "feathers",
        "fins",
        "fur",
        "hair",
        "horns",
        # "scales",
        "wings",
        "underwater",
    ]

    with open("wiki_unigram_dont_delete.pkl", "rb") as f:
        wiki_unigram = pickle.load(f)

    accuracy_by_animal = dict()
    for idx, df_pair in enumerate(df_pairs):
        animal_group_A = pd.read_csv(df_pair[0])
        animal_group_B = pd.read_csv(df_pair[1])
        combine_df = pd.concat([animal_group_A, animal_group_B], axis=0)
        for row in combine_df.iterrows():
            row = row[1]
            animal, curr_acc = row["animal"], row["accuracy"]
            if curr_acc == 1 or curr_acc == 0:
                if animal not in accuracy_by_animal:
                    accuracy_by_animal[animal] = {"accuracy": []}
                    accuracy_by_animal[animal]["wiki_count"] = wiki_unigram[animal] if animal in wiki_unigram else 0
                accuracy_by_animal[animal]["accuracy"].append(curr_acc == 1)

    color = []
    for animal in accuracy_by_animal:
        print(f"animal={animal} number of data points={len(accuracy_by_animal[animal]['accuracy'])}")
        color.append(len(accuracy_by_animal[animal]['accuracy']))
        accuracy_by_animal[animal]["accuracy"] = np.mean(accuracy_by_animal[animal]["accuracy"])
    color = np.array(color)
    animals = accuracy_by_animal.keys()
    X = np.array([accuracy_by_animal[animal]["wiki_count"] for animal in animals])
    Y = np.array([accuracy_by_animal[animal]["accuracy"] for animal in animals])

    x_ticks = np.arange(0, 160000+1, 160000 // 4)
    x_ticks_labels = [f"{tick // 1000}K" for tick in x_ticks]
    A = np.hstack([np.ones((len(X), 1)), X.reshape((-1, 1))])
    b = Y.reshape((-1, 1))
    res = np.linalg.lstsq(a=A, b=b, rcond=-1)
    a, b = res[0][0], res[0][1]
    plt.plot(X, X*b + a, "--", color="red", label="linear regression")
    plt.scatter(X, Y, c=color, alpha=0.8)
    plt.colorbar()
    plt.legend()
    plt.xticks(x_ticks, x_ticks_labels)
    plt.xlabel("Animal Occurrence Count in Wikipedia")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

plot_occurrence_by_animal()
plot_occurrence_by_property(exact=True)
plot_cooccurrence()