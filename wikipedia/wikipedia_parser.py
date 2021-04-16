import re
import xml.etree.ElementTree as ET
from collections import Counter
import os
import threading
import numpy as np
import pickle
import time
import pandas as pd

def find_word_in_text(word, text):
    indices = set()
    indices.add(text.find(f" {word} "))
    indices.add(text.find(f" {word}."))
    indices.add(text.find(f" {word}s "))
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


def compute_unigram():
    jobs = []
    for path in os.walk("."):
        dir, files = path[0], path[-1]
        if 'wiki_00' in files:
            jobs += [os.path.join(dir, file) for file in files]

    threads = []
    jobs_batch = []
    num_of_threads = 20
    batch_size = int(max(np.ceil(len(jobs) / num_of_threads), 1))
    for i in range(0, len(jobs), batch_size):
        jobs_batch.append(jobs[i: min(i + batch_size, len(jobs))])

    results = [None] * len(jobs)

    def process_job(xml_paths, idx):
        print(f"Thread {idx} Start total jobs {len(xml_paths)}")
        for page_num, xml_path in enumerate(xml_paths):
            with open(xml_path, "r") as f:
                data_as_str = f.read()
                time.sleep(1)

            data_as_str = "<top>" + data_as_str + "</top>"
            root = ET.fromstring(data_as_str)
            counter = Counter()

            for i, paper in enumerate(root):
                words = [word.lower() for word in re.sub('[\'\!\?\"\.\,\;\:\(\)\[\]\n]', ' ', paper.text).split(" ") if word]
                curr_counter = Counter(words)
                counter.update(curr_counter)

            if (page_num % 25) == 0:
                print(f"Thread_{idx} processed {page_num} pages.\n")

            if results[idx] is not None:
                results[idx].update(counter)
                time.sleep(1)
            else:
                results[idx] = counter

    print(f"Batch size: {batch_size}")
    print(f"total_docs: {batch_size * num_of_threads}")
    print(f"job queue len={len(jobs_batch)}")
    time.sleep(5)

    for idx, job in enumerate(jobs_batch):
        print(idx)
        thread = threading.Thread(target=process_job, args=(job, idx))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Combine results")
    final_counter = Counter()
    for counter in results:
        final_counter.update(counter)

    print("Done")
    with open("wiki_unigram.pkl", "wb") as f:
        pickle.dump(final_counter, f)


def collect_sentences_with_words(xml_paths, thread_idx, words, sent_length=512):
    sentences = { word: [] for word in words }
    print(f"Thread_{thread_idx} Start")
    for xml_idx, xml_path in enumerate(xml_paths):
        try:
            with open(xml_path, "r") as f:
                data_as_str = f.read()
            data_as_str = "<top>" + data_as_str + "</top>"

            root = ET.fromstring(data_as_str)
        except Exception as e:
            print("error in ", xml_path)
            continue

        for i, paper in enumerate(root):
            orig_text = paper.text.lower()
            for word in words:
                text = orig_text
                idx = find_word_in_text(word, text)
                while idx != -1:
                    chunk = text[max(0, idx - sent_length): min(len(text), idx + sent_length)].lower()
                    text = text[idx + len(word):]
                    idx = find_word_in_text(word, text)
                    sentences[word].append(chunk)

        if xml_idx % 10 == 0:
            print(f"Thread {thread_idx} processed {xml_idx} papers")

    with open(f"thread_{thread_idx}_chunks.pkl", "wb") as f:
        pickle.dump(sentences, f)

    print(f"Thread_{thread_idx} Done")


def collect_all_entities():
    files = ["animals_have_a_beak", "animals_have_horns", "animals_have_fins", "animals_have_scales",
             "animals_have_wings", "animals_have_feathers", "animals_have_fur",
             "animals_have_hair", "animals_live_underwater", "animals_can_fly",
             "animals_dont_have_a_beak", "animals_dont_have_horns", "animals_dont_have_fins",
             "animals_dont_have_scales",
             "animals_dont_have_wings", "animals_dont_have_feathers", "animals_dont_have_fur",
             "animals_dont_have_hair", "animals_dont_live_underwater", "animals_cant_fly", "sanity"]

    entities = []
    for f in files:
        df = pd.read_csv(f"../csv/{f}.csv")
        entities += list(df.entity.values)
    return set(entities)


def run_collect_sentences_with_words():
    jobs = []
    for path in os.walk("."):
        dir, files = path[0], path[-1]
        if 'wiki_00' in files:
            jobs += [os.path.join(dir, file) for file in files]

    threads = []
    jobs_batch = []
    num_of_threads = 20
    batch_size = int(max(np.ceil(len(jobs) / num_of_threads), 1))
    for i in range(0, len(jobs), batch_size):
        jobs_batch.append(jobs[i: min(i + batch_size, len(jobs))])

    print(f"Batch size: {batch_size}")
    print(f"total_docs: {batch_size * num_of_threads}")
    print(f"job queue len={len(jobs_batch)}")
    time.sleep(5)
    words = collect_all_entities()
    words = words.union({"fur", "hair", "water", "underwater", "feather", "wing", "fly", "horn", "scale", "fin", "beak"})
    print(f"words {words}")
    print("len(words)", len(words))

    for thread_idx, job in enumerate(jobs_batch):
        thread = threading.Thread(target=collect_sentences_with_words, args=(job, thread_idx, words))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All Done")


if __name__ == "__main__":
    run_collect_sentences_with_words()
    # compute_unigram()