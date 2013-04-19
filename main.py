import os
import pickle
import nltk
from itertools import tee, izip
from nltk.corpus import wordnet_ic
from semsim import SimilarityCalculator, preprocess

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def preprocess_lyrics(fname):
    pickled_fname = fname + ".processed"
    if os.path.exists(pickled_fname):
        return pickle.load(open(pickled_fname, 'r'))
    else:
        lyrics = open(fname, 'r').read().split('\n')
        lyrics = [preprocess(lyric) for lyric in lyrics]
        pickle.dump(lyrics, open(pickled_fname, 'w'))
        return lyrics

def print_report(f, scores, lyrics, pf):
    results = list(zip(pairwise(lyrics), scores))
    pickle.dump(results, pf)
    for (lyric1, lyric2), score in results:
        f.write('{}\n{}\nScore: {}\n\n'.format(lyric1, lyric2, score))
    f.write('\n')
    threshold = 0.5
    for (lyric1, lyric2), score in results:
        if score > threshold:
            f.write('{}\n{}\n'.format(lyric1, lyric2))
    
def main(fname):
    lyrics = preprocess_lyrics(fname)
    collection = nltk.TextCollection(nltk.corpus.brown)
    ic = wordnet_ic.ic('ic-brown.dat')
    scores = []
    for similarity in SimilarityCalculator.SIMILARITIES.keys():
        output_fname = os.path.join('output', similarity + '.txt')
        if os.path.exists(output_fname):
            continue
        sc = SimilarityCalculator(collection, similarity, ic)
        for lyric1, lyric2 in pairwise(lyrics):
            scores.append(sc.similarity_bidirectional(lyric1, lyric2))
        print_report(open(output_fname, 'w'), scores,
                     open(fname, 'r').read().split('\n'),
                     open(output_fname + '.pickled', 'w'))
    

if __name__ == "__main__":
    main("longestpoem.txt")
    

