import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10
import itertools

class Segment:

    def __init__(self, Pw_unigram, Pw_bigram, Pw_trigram, n_gram=1, smooth_prob = False, stupid_backoff=True, jm_lambda=0.77, interp_lambda3=0.7, interp_lambda2=0.05):
        self.Pw_unigram = Pw_unigram
        self.Pw_bigram = Pw_bigram
        self.Pw_trigram = Pw_trigram
        self.n_gram = n_gram
        self.smooth_prob = smooth_prob
        self.stupid_backoff = stupid_backoff
        self.jm_lambda = jm_lambda
        self.interp_lambda3 = interp_lambda3
        self.interp_lambda2 = interp_lambda2

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []
        return self.iterator_segment(text)

    def trigram_prob_stupid_backoff(self, prev_word1, prev_word2, current_word):
        trigram = "{0} {1} {2}".format(prev_word1, prev_word2, current_word)
        bigram_prev = "{0} {1}".format(prev_word1, prev_word2)
        bigram = "{0} {1}".format(prev_word2, current_word)

        if trigram in self.Pw_trigram and bigram_prev in self.Pw_bigram:
            return math.log(self.Pw_trigram(trigram)) - math.log(self.Pw_bigram(bigram_prev))
        elif bigram in self.Pw_bigram and prev_word2 in self.Pw_unigram: # else backoff to bigrams
            return math.log(0.4) + math.log(self.Pw_bigram(bigram)) - math.log(self.Pw_unigram(prev_word2))
        elif current_word in self.Pw_unigram: # else backoff to unigram
            return math.log(0.4) + math.log(0.4) + math.log(self.Pw_unigram(current_word))
        else:
            return math.log(10. / (self.Pw_unigram.N * 9100 ** len(current_word)))

    def trigram_prob_interpolation(self, prev_word1, prev_word2, current_word):
        trigram = "{0} {1} {2}".format(prev_word1, prev_word2, current_word)
        bigram_prev = "{0} {1}".format(prev_word1, prev_word2)
        bigram = "{0} {1}".format(prev_word2, current_word)
        interp_lambda3 = self.interp_lambda3
        interp_lambda2 = self.interp_lambda2
        interp_lambda1 = 1 - interp_lambda3 - interp_lambda2

        trigram_prob = 0
        bigram_prob = 0
        unigram_prob = 0

        if trigram in self.Pw_trigram and bigram_prev in self.Pw_bigram:
            trigram_prob = math.log(self.Pw_trigram(trigram) / self.Pw_bigram(bigram_prev))

        if bigram in self.Pw_bigram and prev_word2 in self.Pw_unigram:
            bigram_prob = math.log(self.Pw_bigram(bigram) / self.Pw_unigram(prev_word2))

        if current_word in self.Pw_unigram:
            unigram_prob = math.log(self.Pw_unigram(current_word))

        if trigram_prob is not 0 and bigram_prob is not 0 and unigram_prob is not 0:
            return math.log(math.exp(math.log(interp_lambda3) + trigram_prob) + math.exp(math.log(interp_lambda2) + bigram_prob) + math.exp(math.log(interp_lambda1) + unigram_prob))
        elif bigram_prob is not 0 and unigram_prob is not 0:
            return math.log(math.exp(math.log(interp_lambda2) + bigram_prob) + math.exp(math.log(interp_lambda1) + unigram_prob))
        elif unigram_prob is not 0:
            return math.log(interp_lambda1) + unigram_prob
        else:
            return math.log(10. / (self.Pw_unigram.N * 9100 ** len(current_word)))

    def bigram_prob_stupid_backoff(self, prev_word, current_word):
        bigram = "{0} {1}".format(prev_word, current_word)

        # if combination of 'prev current' is in the bigrams else backoff to unigram
        if bigram in self.Pw_bigram and prev_word in self.Pw_unigram:
            return math.log(self.Pw_bigram(bigram) / self.Pw_unigram(prev_word))
        elif current_word in self.Pw_unigram:  # else back off to unigram
            return math.log(0.4) + math.log(self.Pw_unigram(current_word))
        else:
            return math.log(self.Pw_unigram(current_word))

    def bigram_prob_jm_smoothing(self, prev_word, current_word):
        jm_lambda = self.jm_lambda
        bigram = "{0} {1}".format(prev_word, current_word)

        bigram_prob = 0
        unigram_prob = 0

        if bigram in self.Pw_bigram and prev_word in self.Pw_unigram:
            bigram_prob = math.log(self.Pw_bigram(bigram) / self.Pw_unigram(prev_word))

        if current_word in self.Pw_unigram:
            unigram_prob = math.log(self.Pw_unigram(current_word))

        if bigram_prob is not 0 and unigram_prob is not 0:
            return math.log(math.exp(math.log(jm_lambda) + bigram_prob) + math.exp(math.log(1 - jm_lambda) + unigram_prob))
        elif unigram_prob is not 0:
            return math.log(1 - jm_lambda) + unigram_prob
        else:
            return math.log(10./(self.Pw_unigram.N * 9100**len(current_word)))

    def iterator_segment(self, input, L=8):
        chart = defaultdict(int)
        hq = []

        ## Initialize the heap ## - for each word that matches input at position 0 - inserting into heap
        for i in range(min(len(input), L)):
            word = input[:i+1]

            if self.n_gram is 3:  # [TRIGRAM MODEL] - Form `prev_word1 prev_word2 word`
                if self.smooth_prob: # linear interpolation
                    log_prob = self.trigram_prob_interpolation(prev_word1="<S>", prev_word2="<S>", current_word=word)
                else:  # stupid back-off
                    log_prob = self.trigram_prob_stupid_backoff(prev_word1="<S>", prev_word2="<S>", current_word=word)

            elif self.n_gram is 2:  # [BIGRAM MODEL] Form `prev_word word`
                if self.smooth_prob: # jelinek-mercer smoothing
                    log_prob = self.bigram_prob_jm_smoothing(prev_word="<S>", current_word=word)
                else:  # stupid back-off
                    log_prob = self.bigram_prob_stupid_backoff(prev_word="<S>", current_word=word)

            else:  # [UNIGRAM MODEL]
                log_prob = math.log(self.Pw_unigram(word))

            entry = (log_prob, word, len(word), None)
            heapq.heappush(hq, entry)

        # until heap is not empty
        while hq:
            # fetching the top entry in heap
            heapq._heapify_max(hq)
            entry = heapq.heappop(hq)

            end_index = entry[2] - 1 # end_index is the length of the entry word

            # if chart[end_index] has a previous entry at end_index
            if chart[end_index] != 0:
                # if entry has a higher log prob then chart[end_index]
                # then put entry into chart[end_index], continue otherwise
                if entry[0] > chart[end_index][0]:
                    chart[end_index] = entry
                else:
                    continue
            else:
                chart[end_index] = entry

            # get newwords that matches input starting at position endindex+1
            for i in range(end_index+1, end_index+1 + min(len(input[end_index+1:]), L)):
                new_word = input[end_index+1:i+1]

                if self.n_gram is 3:  # [TRIGRAM MODEL] Form `prev_word1 prev_word2 new_word`
                    if chart[end_index][3] is not None:
                        prev_word2 = chart[end_index][1]
                        if chart[chart[end_index][3]][3] is not None:
                            prev_word1 = chart[chart[end_index][3]][1]
                        else:  # if end_index is None it means the previous word should be <S> start of sentence
                            prev_word1 = "<S>"
                    else:  # if end_index is None it means the previous word should be <S> and before it definitly is <S>
                        prev_word1 = "<S>"
                        prev_word2 = "<S>"

                    if self.smooth_prob:  # [BIGRAM MODEL] linear interpolation
                        log_prob = self.trigram_prob_interpolation(prev_word1=prev_word1, prev_word2=prev_word2, current_word=new_word)
                    else:  # stupid back-off
                        log_prob = self.trigram_prob_stupid_backoff(prev_word1=prev_word1, prev_word2=prev_word2, current_word=new_word)

                elif self.n_gram is 2:  # [UNIGRAM MODEL] Form `prev_word new_word`
                    if chart[end_index][3] is not None:
                        prev_word = chart[end_index][1]
                    else:  # if end_index is None it means the previous word should be <S> start of sentence
                        prev_word = "<S>"

                    if self.smooth_prob:  # jelinek-mercer smoothing
                        log_prob = self.bigram_prob_jm_smoothing(prev_word=prev_word, current_word=new_word)
                    else:  # stupid back-off
                        log_prob = self.bigram_prob_stupid_backoff(prev_word=prev_word, current_word=new_word)

                else:
                    log_prob = math.log(self.Pw_unigram(new_word))

                new_entry = (entry[0] + log_prob, new_word, entry[2] + len(new_word), end_index)

                # if new_entry doesnt exist in heap add it
                if new_entry not in hq:
                    heapq.heappush(hq, new_entry)

        # Best segmentation
        segments = []
        idx = len(input) - 1
        while idx is not None:
            segments.append(chart[idx][1])
            idx = chart[idx][3]

        # since segments are in reverse order reverse them to correct order
        # before returning
        return segments[::-1]

#### Support functions (p. 224)
class Pdist_unigram(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key):
        if key == "<S>": return 0.5
        if key in self: return self[key]/self.N
        else: return self.missingfn(key, self.N)

class Pdist_bigram(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None):
        for key, count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
    def __call__(self, key):
        if key is "<S> <S>": return 0.5
        if key in self:
            return self[key] / self.N

class Pdist_trigram(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None):
        for key, count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
    def __call__(self, key):
        if key in self:
            return self[key] / self.N

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

def avoid_long_words(word, N):
    "Estimate the probability of an unknown word."
    return 10./(N * 9100**len(word))

def extract_trigrams():
    trigrams = defaultdict(int)

    with open("data/train.txt") as f:
        for line in f:
            sentence = "<S> <S> " + line.strip()

            tokens = sentence.split(" ")

            sentence_trigrams = zip(*[tokens[i:] for i in range(3)])
            sentence_trigrams = [" ".join(ngram) for ngram in sentence_trigrams]
            for sentence_trigram in sentence_trigrams:
                trigrams[sentence_trigram] += 1

    f = open("data/count_3w.txt", "w")
    for key, val in trigrams.items():
        f.write("{0}\t{1}\n".format(key, val))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-t", "--trigramcounts", dest='counts3w', default=os.path.join('data', 'count_3w.txt'), help="trigram counts [default: data/count_3w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    optparser.add_option("-m", "--model", dest="model_version", default=0, help="log file for debugging", type=int)

    (opts, _) = optparser.parse_args()

    opts.model_version = 4

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    if opts.model_version == 0: # unigram
        n_gram=1
        smooth_prob = False
    elif opts.model_version == 1: # bigram with stupid-backoff
        n_gram=2
        smooth_prob = False
    elif opts.model_version == 2: # bigram with jelinek-mercer smoothing
        n_gram=2
        smooth_prob = True
    elif opts.model_version == 3: # trigram with stupid-backoff
        n_gram=3
        smooth_prob = False
    elif opts.model_version == 4: # trigram with linear interpolation smoothing
        n_gram=3
        smooth_prob = True

    # Extract trigrams file `data/count_3w.txt` if selected model is trigram one
    if opts.model_version >= 3:
        # Extract trigrams if the file `data/count_3w.txt` doesn't exist
        if not os.path.exists('data/count_3w.txt'):
            # extract the `data/train.txt.bz2`
            if not os.path.exists('data/train.txt'):
                import bz2

                zipfile = bz2.BZ2File('data/train.txt.bz2')
                data = zipfile.read()
                open('data/train.txt', 'wb').write(data)

            # extract trigrams to `data/count_3w.txt`
            extract_trigrams()

    Pw_unigram = Pdist_unigram(data=datafile(opts.counts1w), missingfn=avoid_long_words)
    Pw_bigram = Pdist_bigram(data=datafile(opts.counts2w))
    Pw_trigram = Pdist_trigram(data=datafile(opts.counts3w))
    segmenter = Segment(Pw_unigram, Pw_bigram, Pw_trigram, n_gram=n_gram, smooth_prob=smooth_prob)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
