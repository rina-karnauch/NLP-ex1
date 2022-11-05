import numpy
from more_itertools import pairwise
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
        if counter > 20:
          break
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
        wi_0, wi_1 = pair[0], pair[1]
        wi_count = first_lemma_pair_dict[wi_0]
        probability_dict[pair] = count / wi_count
    return probability_dict


def get_all_pairs(documents):
    all_words = []
    start_pairs = []
    for doc in documents:
        gen = [token for token in doc if token.is_alpha]
        if gen:
            start_pairs.append(("START", gen[0].lemma_))
        for token in gen:
            lemma = token.lemma_
            all_words.append(lemma)
    return list(pairwise(all_words)) + start_pairs


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


if __name__ == '__main__':
    str_test1 = "I have a house in"

    documents = data_parser()
    probability_dict = train_bigram(documents)

    max_value = 0
    last_word = str_test1.split()[-1]
    max_pair = ""
    for pair, prob in probability_dict.items():
        p0, p1 = pair[0], pair[1]
        if p0 == last_word and prob > max_value:
            max_pair = pair
            max_value = prob
    print(max_pair)

    str_test2 = "START Brad Pitt was born in Oklahoma"
    str_test2_words = str_test2.split()
    str_test3 = "START The actor was born in USA"
    str_test3_words = str_test3.split()

    prob_2, prob_3 = 1, 1
    perplexity_2, perplexity_3 = 0, 0

    for word0, word1 in zip(str_test2_words, str_test2_words[1:]):
        if (word0, word1) in probability_dict.keys():
            prob_current = probability_dict[(word0, word1)]
        else:
            prob_current = 0
        prob_2 = prob_2 * prob_current
        perplexity_2 += -1 * numpy.log(prob_current)
    print("probability_2 is:", prob_2)
    print("perplexity_2 is:", perplexity_2)

    for word0, word1 in zip(str_test3_words, str_test3_words[1:]):
        if (word0, word1) in probability_dict.keys():
            prob_current = probability_dict[(word0, word1)]
        else:
            prob_current = 0
        prob_3 = prob_3 * prob_current
        perplexity_3 += -1 * numpy.log(prob_current)
    print("probability_3 is:", prob_3)
    print("perplexity_3 is:", perplexity_3)

