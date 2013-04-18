import string
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

def preprocess(text):
    """
    Helper function to preprocess text (lowercase, remove punctuation etc.)
    """
    text = text.translate(None, string.punctuation)
    words = filter(None, re.split('\s+', text))
    words = nltk.pos_tag(words)
    words = [(word, nltk.simplify_wsj_tag(tag)) for word, tag in words]
    words = [(word, 'V') if tag.startswith('V') else (word, tag)
             for word, tag in words]
    return words

class SimilarityCalculator:

    SIMILARITIES = {'path' : lambda s1, s2, ic: s1.path_similarity(s2),
                    'lch' : lambda s1, s2, ic: s1.lch_similarity(s2),
                    'wup' : lambda s1, s2, ic: s1.wup_similarity(s2),
                    'res' : lambda s1, s2, ic: s1.res_similarity(s2, ic),
                    'jcn' : lambda s1, s2, ic: s1.jcn_similarity(s2, ic),
                    'lin' : lambda s1, s2, ic: s1.path_similarity(s2, ic)}

    def __init__(self, collection, sim_measure, ic):
        self.pos_tags = ['N', 'ADJ', 'V', 'ADV']
        self.wn_pos_tags = [wn.NOUN, wn.ADJ, wn.VERB, wn.ADV]
        self.collection = collection
        self.sim_measure = SimilarityCalculator.SIMILARITIES[sim_measure]
        self.ic = ic

    def word_similarity(self, word1, word2, pos_tag):
        pos_tag = self.wn_pos_tags[self.pos_tags.index(pos_tag)]
        synsets1 = wn.synsets(word1, pos_tag)
        synsets2 = wn.synsets(word2, pos_tag)
        return max(self.sim_measure(sense1, sense2, self.ic)
                   for sense1 in synsets1 for sense2 in synsets2)

    def max_similarity(self, word, words, pos_tag):
        return max(self.word_similarity(word, other, pos_tag)
                   for other in words) or 0

    def similarity(self, *sentences):
        pos_sets = {}
        for tag in self.pos_tags:
            pos_sets[tag] = [[word for word, pos_tag in sentence
                              if pos_tag == tag] for sentence in sentences]
        idf = {}
        for tag in self.pos_tags:
            for word in pos_sets[tag][0]:
                idf[word] = self.collection.idf(word)
        sim = sum(sum(self.max_similarity(word, pos_sets[tag][1], tag) * idf[word]
                      for word in pos_sets[tag][0]) for tag in self.pos_tags)
        sim /= sum(idf.values())
        return sim

def test():
    col = nltk.TextCollection(["Have you ever had a dream at night, and remembered it years later?",
                              "Ok Earth Wind and Fire was cool but Carly Simon?..... Waiter!!"
                              "Busy busy week. Finals due next week and tomorrow!",
                              "Just did a trust fall onto the snow. It saved me. <3"])
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    sc = SimilarityCalculator(col, 'lin', brown_ic)
    sentence1 = preprocess("Just did a trust fall onto the snow. It saved me. <3")
    sentence2 = preprocess("Why have changed a lot. I really can not believe what I see ??")
    print sc.similarity(sentence1, sentence2)

if __name__ == "__main__":
    test()
