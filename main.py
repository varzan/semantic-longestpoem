import os
import pickle
import datetime
import nltk
from itertools import tee, izip
from nltk.corpus import wordnet_ic
from semsim import SimilarityCalculator, preprocess

def pairwise(iterable):
    return izip(iterable[::2], iterable[1::2])

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
    for threshold in range(20, 85, 5):
        float_thresh = threshold / 100.0
        selected = [(lyric1, lyric2) for (lyric1, lyric2), score in results
                    if score > float_thresh]
        nselected = len(selected) * 2
        ntotal = len(lyrics)
        ratio = nselected/float(ntotal)
        f.write('Threshold = {}\n'.format(float_thresh))
        f.write('Selected {} lyrics out of {} ({} %)\n\n'.
                format(nselected , ntotal, ratio * 100))
        for lyric1, lyric2 in selected:
            f.write('{}\n{}\n'.format(lyric1, lyric2))
        f.write('\n')
    
def main(fname):
    lyrics = preprocess_lyrics(fname)
    collection = nltk.TextCollection(nltk.corpus.brown)
    ic = wordnet_ic.ic('ic-brown.dat')
    scores = []
    for similarity in SimilarityCalculator.SIMILARITIES.keys():
        output_fname = os.path.join('output', similarity + '.txt')
        pickled_fname = output_fname + '.pickled'
        if os.path.exists(output_fname):
            continue
        now = datetime.datetime.now()
        print '[{}] Starting calculation on {}'.format(str(now), similarity)
        if os.path.exists(pickled_fname):
            scores = [score for couplet, score in
                      pickle.load(open(pickled_fname, 'r'))]
        else:              
            sc = SimilarityCalculator(collection, similarity, ic)
            for lyric1, lyric2 in pairwise(lyrics):
                scores.append(sc.similarity_bidirectional(lyric1, lyric2))
        print_report(open(output_fname, 'w'), scores,
                     open(fname, 'r').read().split('\n'),
                     open(pickled_fname, 'w'))
        now = datetime.datetime.now()
        print '[{}]: Finished calculation on {}'.format(str(now), similarity)
    

if __name__ == "__main__":
    main("longestpoem.txt")
    

