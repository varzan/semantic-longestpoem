import os
import pickle
import datetime
import nltk
import matplotlib.pyplot as plt
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

def print_report(f, scores, lyrics, pf, img_fname):
    results = list(zip(pairwise(lyrics), scores))
    pickle.dump(results, pf)
    for (lyric1, lyric2), score in results:
        f.write('{}\n{}\nScore: {}\n\n'.format(lyric1, lyric2, score))
    f.write('\n')
    thresh_scores = {}
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
        thresh_scores[float_thresh] = nselected
    plt.clf()
    plt.hist(scores, bins = 20, range=(0.0, 1.0))
    plt.savefig(img_fname)
    f.close()
    pf.close()
    return thresh_scores
    
def main(fname):
    lyrics = preprocess_lyrics(fname)
    collection = nltk.TextCollection(nltk.corpus.brown)
    ic = wordnet_ic.ic('ic-brown.dat')
    thresh_counts = {}
    for similarity in SimilarityCalculator.SIMILARITIES.keys() + ['bp_adj']:
        scores = []
        output_fname = os.path.join('output', similarity + '.txt')
        pickled_fname = output_fname + '.pickled'
        img_fname = os.path.join('output', similarity + '_hist.png')
        if os.path.exists(output_fname):
            continue
        now = datetime.datetime.now()
        print '[{}] Starting calculation on {}'.format(str(now), similarity)
        if similarity == 'bp':
            adjust_bp()
        if os.path.exists(pickled_fname):
            scores = [score for couplet, score in
                      pickle.load(open(pickled_fname, 'r'))]
        else:              
            sc = SimilarityCalculator(collection, similarity, ic)
            for lyric1, lyric2 in pairwise(lyrics):
                scores.append(sc.similarity_bidirectional(lyric1, lyric2))
        thresh_counts[similarity] = print_report(open(output_fname, 'w'),
                                                 scores,
                                                 open(fname, 'r').read().
                                                 split('\n'),
                                                 open(pickled_fname, 'w'),
                                                 img_fname)
        now = datetime.datetime.now()
        print '[{}] Finished calculation on {}'.format(str(now), similarity)
    plt.clf()
    for similarity in thresh_counts.keys():
        res = list(thresh_counts[similarity].iteritems())
        res.sort()
        res = zip(*res)
        plt.plot(res[0], res[1], label=similarity, zorder=1)
        plt.scatter(res[0], res[1], zorder=2)
    plt.legend()
    plt.xlabel("threshold")
    plt.ylabel("no. lyrics selected")
    plt.savefig(os.path.join("output", "thresholds.png"))
    
def adjust_bp():
    bp_fname = os.path.join('output', 'bp.txt.pickled')
    bp_adj_fname = os.path.join('output', 'bp_adj.txt.pickled')
    results = pickle.load(open(bp_fname, 'r'))
    scores = [score for couplet, score in results]
    couplets = [couplet for couplet, score in results]
    scores = [score * 4 for score in scores]
    pickle.dump(zip(couplets, scores), open(bp_adj_fname, 'w'))
    
if __name__ == "__main__":
    main("longestpoem.txt")
    

