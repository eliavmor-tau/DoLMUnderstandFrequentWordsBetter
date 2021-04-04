import re
import xml.etree.ElementTree as ET
from collections import Counter
import os
import threading
import numpy as np
import pickle

jobs = []
for path in os.walk("."):
    dir, files = path[0], path[-1]
    if 'wiki_00' in files:
        jobs += [os.path.join(dir, file) for file in files]

threads = []
jobs_batch = []
num_of_threads = 20
batch_size = int(max(np.ceil(len(jobs) / 20), 1))
for i in range(0, len(jobs), batch_size):
    jobs_batch.append(jobs[i: min(i + batch_size, len(jobs))])

results = [None] * len(jobs)


def process_job(xml_path, idx):
    with open(xml_path, "r") as f:
        data_as_str = f.read()
    data_as_str = "<top>" + data_as_str + "</top>"

    root = ET.fromstring(data_as_str)
    counter = Counter()
    print(f"Thread {idx} Start")
    for i, paper in enumerate(root):
        words = [word.lower() for word in re.sub('[\.\,\;\:\(\)\[\]\n]', ' ', paper.text).split(" ") if word]
        curr_counter = Counter(words)
        counter.update(curr_counter)
        if i % 1000 == 0:
            print(f"Thread {idx} processed {i} papers.\n")
    results[idx] = counter

print(f"Batch size: {batch_size}")

for idx, job in enumerate(jobs):
    thread = threading.Thread(target=process_job, args=(job, idx))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

final_counter = Counter()
for counter in results:
    final_counter.update(counter)

print("Done")
with open("wiki_unigram.pkl", "wb") as f:
    pickle.dump(final_counter, f)

