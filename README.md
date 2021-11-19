# Do Language Models Understand Frequnt Words Better?

This work was inspired by Talmor's work [1] on what LM pretraining capture.
A report of my work can be found here: https://github.com/eliavmor-tau/DoLMUnderstandFrequentWordsBetter/blob/main/Do%20Language%20Models%20Understand%20Frequent%20Word%20Better.pdf

## Abstract 
Pre-trained language models (LMs) have recently demonstrated outstanding results across a variety of tasks. However, it remains unclear precisely what knowledge the LM manages to capture during pre-training and how word frequency in the training corpus affects the acquisition of knowledge about these words.
In this work we explore the correlation between word occurrence in the language and general world knowledge acquired by a Language Model during pre-training. We propose a framework for testing this subject, using a downstream "Yes/No" QA task. The model (in our case T5) is trained on a large set of questions and then tested on its knowledge of relations between different animals and their properties, e.g. "Does a fish have wings?". Our findings show positive correlations between: (a) word occurrence and the accuracy of answers for this word. (b) co-occurrence of two words and a tendency of the model to answer "Yes" for questions about their relation. We also created a classifier for predicting the answer of the model for animal/property relation questions based on the co-occurrence rate of these two words.

## References
<a id="1">[1]</a> 
oLMpics -- On what Language Model Pre-training Captures,
Alon Talmor and Yanai Elazar and Yoav Goldberg and Jonathan Berant (2020).
https://arxiv.org/abs/1912.13283
