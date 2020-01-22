import os, sys, optparse
import tqdm
import re
import numpy as np
import pymagnitude
from collections import defaultdict

isNumber = re.compile(r'\d+.*')
def norm_word(word):
    if isNumber.search(word.lower()):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word.lower()


def read_lexicon(filename):
    lexicon = {}
    for line in open(filename, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon


def edge_vectors(wvecs, word, ontology, dim):
    
    #check if any synonym exists in lexicon for the given word vector
    if word not in ontology.keys() or len(ontology[word]) == 0:
        return (np.zeros(dim),0)
    else:
        # consider beta = (degree of node)^-1
        beta = 1 / len(ontology[word])

        sum1 = np.zeros(dim)
        # for every synonym located in lexicon for the word vector of a given word
        # bring the similar words closer in the retrofitted vector
        for similar_word in ontology[word]:
            if norm_word(similar_word) != '---num---' and norm_word(similar_word) != '---punc---' and similar_word in wvecs.keys():
                sum1 += beta * wvecs[similar_word]
        
        sum2 = beta * len(ontology[word])

        return (sum1, sum2)


def main(opts):
    
    wvecs = pymagnitude.Magnitude(opts.wordvecfile)
    dim = wvecs.dim
    word_vecs = {}
    
    #copy the read-only word vectors in a dictionary for modification
    for k in tqdm.trange(len(wvecs)):
        word_vecs[wvecs[k][0]] = wvecs[k][1]
    
    alpha = 0.075
    #normalizing the lexicon to all lower case letters
    ontology = read_lexicon(opts.lexicon)
    
    for t in tqdm.trange(10):
        for i in range(len(wvecs)):
            
            wordvec = wvecs[i]	
            #calculate the retrofitted matrix by increasing the similarities between the synonyms as per the lexicon file
            sums = edge_vectors(word_vecs, wordvec[0], ontology, dim)
            word_vecs[wordvec[0]] = (sums[0] + alpha*wordvec[1]) / (sums[1] + alpha)

    f = open(opts.output, 'w', encoding = 'utf-8')
    
    #write the retrofitted word vectors in output file
    for k,v in word_vecs.items():
        f.write(str(k)
        for x in v:
            f.write(' ' + str(x))
        f.write('\n')

    f.close()


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-l", "--lexicon", dest="lexicon", default=os.path.join('data', 'lexicons', 'wordnet-synonyms.txt'), help="lexicon path")
    optparser.add_option("-o", "--output", dest="output", default=os.path.join('data', 'glove.6B.100d.retrofit.txt'), help="output txt file path")
    (opts, _) = optparser.parse_args()

    main(opts)