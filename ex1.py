import spacy
import numpy
import math
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")
START_TOKENS = nlp("START")
START_LEMMA = START_TOKENS[0].lemma_


def data_parser():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    documents = []
    for text in dataset['text']:
        doc = nlp(text)
        lemmas = doc_to_lemmas(doc)
        if lemmas:
            documents.append(lemmas)
    return documents


def doc_to_lemmas(doc):
    doc_words = []
    gen = [token for token in doc if token.is_alpha]
    if gen:
        doc_words.append(START_LEMMA)
        for token in gen:
            lemma = token.lemma_
            doc_words.append(lemma)
    return doc_words


def train_unigram(documents):
    word_dict = {}
    count = 0
    for doc in documents:
        for lemma in doc:
            if lemma not in word_dict.keys():
                word_dict[lemma] = 0
            word_dict[lemma] += 1
            count += 1
    prob_dict = {k: v / count for k, v in zip(word_dict.keys(), word_dict.values())}
    return prob_dict


def train_bigram(documents):
    pair_dict = {}
    first_lemma_pair_dict = {}
    for doc in documents:
        for lemma0, lemma1 in zip(doc, doc[1:]):
            if (lemma0, lemma1) not in pair_dict.keys():
                pair_dict[(lemma0, lemma1)] = 0
            pair_dict[(lemma0, lemma1)] += 1
            if lemma0 not in first_lemma_pair_dict.keys():
                first_lemma_pair_dict[lemma0] = 0
            first_lemma_pair_dict[lemma0] += 1
    prob_dict = {}
    for pair, count in pair_dict.items():
        l0 = pair[0]
        w_count = first_lemma_pair_dict[l0]
        prob_dict[pair] = count / w_count
    return prob_dict


def linear_interpolation(test_doc, unigram_dict, bigram_dict):
    g0 = 1 / 3
    g1 = 2 / 3
    prob = 1
    test_lemmas = doc_to_lemmas(test_doc)
    for l0, l1 in zip(test_lemmas, test_lemmas[1:]):
        if l1 in unigram_dict.keys():
            prob_unigram_current = unigram_dict[l1]
        else:
            prob_unigram_current = 0
        if (l0, l1) in bigram_dict.keys():
            prob_bigram_current = bigram_dict[(l0, l1)]
        else:
            prob_bigram_current = 0
        prob = prob * ((g0 * prob_unigram_current) + (g1 * prob_bigram_current))
    return prob


def test_doc(test_doc, prob_dict):
    prob = 1
    test_lemmas = doc_to_lemmas(test_doc)
    for l0, l1 in zip(test_lemmas, test_lemmas[1:]):
        if (l0, l1) in prob_dict.keys():
            prob_current = prob_dict[(l0, l1)]
        else:
            prob_current = 0
        prob = prob * prob_current
    return prob


if __name__ == '__main__':
    documents = data_parser()
    bigram_prob = train_bigram(documents)
    unigram_prob = train_unigram(documents)

    test_doc1 = nlp("I have a house in")
    test_doc2 = nlp("Brad Pitt was born in Oklahoma")
    test_doc3 = nlp("The actor was born in USA")

    test_tokens_amount = len(test_doc2) + len(test_doc3)

    max_value = 0
    max_pair = None
    last_word = test_doc1[-1].lemma_
    for pair, prob in bigram_prob.items():
        p0, p1 = pair[0], pair[1]
        if p0 == last_word and prob > max_value:
            max_pair = pair
            max_value = prob
    print("-------------------------------------------")
    print("most probable word predicted by model: ", max_pair[1])

    prob_2 = test_doc(test_doc2, bigram_prob)
    prob_3 = test_doc(test_doc3, bigram_prob)

    print("-------------------------------------------")
    print("probability for:" + "'Brad Pitt was born in Oklahoma': ", numpy.log(prob_2))
    print("probability for:" + "'The actor was born in USA': ", numpy.log(prob_3))

    perplexity_power = (1/test_tokens_amount) * (numpy.log(prob_2) + numpy.log(prob_3))
    print("perplexity is: ", -1 * math.exp(perplexity_power))

    linear_prob_2 = linear_interpolation(test_doc2, unigram_prob, bigram_prob)
    linear_prob_3 = linear_interpolation(test_doc3, unigram_prob, bigram_prob)

    print("-------------------------------------------")
    print("linear interpolation probability for:" + "'Brad Pitt was born in Oklahoma': ", numpy.log(linear_prob_2))
    print("linear interpolation probability for:" + "'The actor was born in USA': ", numpy.log(linear_prob_3))

    perplexity_power = (1/test_tokens_amount) * (numpy.log(linear_prob_2) + numpy.log(linear_prob_3))
    print("perplexity is: ", math.exp(-1*perplexity_power))
