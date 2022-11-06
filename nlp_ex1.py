import numpy
import spacy
from datasets import load_dataset


def data_parser():
    nlp = spacy.load("en_core_web_sm")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    documents = []
    counter = 0
    for text in dataset['text']:
        counter += 1
        doc = nlp(text)
        documents.append(doc)
    return documents


def train_unigram(documents):
    word_dict = {}
    count = 0
    for doc in documents:
        gen = (token for token in doc if token.is_alpha)
        for token in gen:
            lemma = token.lemma_
            if lemma not in word_dict.keys():
                word_dict[lemma] = 0
            word_dict[lemma] += 1
            count += 1

    prob_dict = {k: v / count for k, v in zip(word_dict.keys(), word_dict.values())}
    return prob_dict


def train_bigram(documents):
    all_pairs = get_all_pairs(documents)
    pair_dict = {}
    first_lemma_pair_dict = {}

    for pair in all_pairs:
        if pair not in pair_dict.keys():
            pair_dict[pair] = 0
        pair_dict[pair] += 1
        if pair[0] not in first_lemma_pair_dict.keys():
            first_lemma_pair_dict[pair[0]] = 0
        first_lemma_pair_dict[pair[0]] += 1

    probability_dict = {}
    for pair, count in pair_dict.items():
        w_0, w_1 = pair[0], pair[1]
        w_count = first_lemma_pair_dict[w_0]
        probability_dict[pair] = count / w_count
    return probability_dict


def get_all_pairs(documents):
    pairs = []
    for doc in documents:
        doc_words = doc_to_lemmas(doc)
        for word0, word1 in zip(doc_words, doc_words[1:]):
            pairs.append((word0, word1))
    return pairs


def linear_interpolation(documents, sentence):
    unigram_dict = train_unigram(documents)
    bigram_dict = train_bigram(documents)

    prob, perplexity = 1, 0

    sentence_words = sentence.split()

    for word0, word1 in zip(sentence_words, sentence_words[1:]):
        if word1 in unigram_dict.keys():
            current_unigram = unigram_dict[word1]
        else:
            current_unigram = 0
        if (word0, word1) in bigram_dict.keys():
            current_bigram = bigram_dict[(word0, word1)]
        else:
            current_bigram = 0
        current_prob = (1 / 3) * current_unigram + (2 / 3) * current_bigram
        perplexity += -1 * numpy.log(current_prob)
        prob *= current_prob

    print("prob for sentence:" + sentence, prob)
    print("perplexity for sentence:" + sentence, perplexity)


def doc_to_lemmas(doc):
    doc_words = []
    gen = [token for token in doc if token.is_alpha]
    if gen:
        doc_words.append("START")
        for token in gen:
            lemma = token.lemma_
            doc_words.append(lemma)
    return doc_words


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")

    documents = data_parser()
    probability_dict = train_bigram(documents)

    test_doc1 = nlp("I have a house in")
    test_doc2 = nlp("Brad Pitt was born in Oklahoma")
    test_doc3 = nlp("The actor was born in USA")

    max_value = 0
    last_word = test_doc1[-1]
    max_pair = ""
    for pair, prob in probability_dict.items():
        p0, p1 = pair[0], pair[1]
        if nlp(p0)[0] == last_word and prob > max_value:
            max_pair = pair
            max_value = prob
    print(max_pair)

    prob_2, prob_3 = 1, 1
    perplexity_2, perplexity_3 = 0, 0

    for word0, word1 in zip(test_doc2, test_doc2[1:]):
        if (word0, word1) in probability_dict.keys():
            prob_current = probability_dict[(word0, word1)]
        else:
            prob_current = 0
            print(word0 + "," + word1)
        prob_2 = prob_2 * prob_current
        perplexity_2 += -1 * numpy.log(prob_current)
    print("probability_2 is:", prob_2)
    print("perplexity_2 is:", perplexity_2)

    for word0, word1 in zip(test_doc3, test_doc3[1:]):
        if (word0, word1) in probability_dict.keys():
            prob_current = probability_dict[(word0, word1)]
        else:
            prob_current = 0
        prob_3 = prob_3 * prob_current
        perplexity_3 += -1 * numpy.log(prob_current)
    print("probability_3 is:", prob_3)
    print("perplexity_3 is:", perplexity_3)
