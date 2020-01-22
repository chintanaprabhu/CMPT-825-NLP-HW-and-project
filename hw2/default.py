import os, sys, optparse
import tqdm
import pymagnitude
import math
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter
import re

stop_words = ["e.g.", "http://pgina.xpasystems.com","'s", "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]


class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    def is_num(self, word):
        return re.compile(r'\d+.*').search(word.lower())

    def is_punctuation(self, word):
        return re.sub(r'\W+', '', word) == ''

    def pcos(self, a, b):
        return (self.wvecs.similarity(a, b) + 1) /2

    def add_similarity(self, target_word, sub_words, sentence):
        similarities = {}
        for sub_word, _ in sub_words:
            sub_context_similarity_sum = 0
            context_words_count = 0
            # print(" ".join(sentence))
            for context_word in sentence:
                if context_word is not target_word and context_word not in stop_words and not self.is_punctuation(context_word) and not self.is_num(context_word):
                    # print(context_word)
                    context_words_count += 1
                    sub_context_similarity_sum += self.wvecs.similarity(sub_word, context_word)
            if context_words_count == 0: context_words_count = 1
            similarities[sub_word] = (self.wvecs.similarity(sub_word, target_word) + sub_context_similarity_sum) / (context_words_count + 1)

        similarities = sorted(similarities.items(), key=itemgetter(1), reverse=True)
        return [similarity[0] for similarity in similarities]

    def baladd_similarity(self, target_word, sub_words, sentence):
        similarities = {}
        for sub_word, _ in sub_words:
            sub_context_similarity_sum = 0
            context_words_count = 0

            for context_word in sentence:
                if context_word is not target_word and context_word not in stop_words and not self.is_punctuation(context_word) and not self.is_num(context_word):
                    context_words_count += 1
                    sub_context_similarity_sum += self.wvecs.similarity(sub_word, context_word)
            if context_words_count == 0: context_words_count = 1
            similarities[sub_word] = (context_words_count * self.wvecs.similarity(sub_word, target_word) + sub_context_similarity_sum) / (2 * context_words_count)

        similarities = sorted(similarities.items(), key=itemgetter(1), reverse=True)
        return [similarity[0] for similarity in similarities]

    def mult_similarity(self, target_word, sub_words, sentence):
        similarities = {}
        for sub_word, _ in sub_words:
            sub_context_similarity_sum = 0
            context_words_count = 0

            for context_word in sentence:
                if context_word is not target_word and context_word not in stop_words and not self.is_punctuation(context_word) and not self.is_num(context_word):
                    context_words_count += 1
                    sub_context_similarity_sum *= self.pcos(self.wvecs.query(sub_word), self.wvecs.query(context_word))
            if context_words_count == 0: context_words_count = 1
            similarities[sub_word] = (self.pcos(sub_word, target_word) * sub_context_similarity_sum) ** (1 / (context_words_count + 1))


        similarities = sorted(similarities.items(), key=itemgetter(1), reverse=True)
        return [similarity[0] for similarity in similarities]

    def balmult_similarity(self, target_word, sub_words, sentence):
        similarities = {}
        for sub_word, _ in sub_words:
            sub_context_similarity_sum = 0
            context_words_count = 0

            for context_word in sentence:
                if context_word is not target_word and context_word not in stop_words and not self.is_punctuation(context_word) and not self.is_num(context_word):
                    context_words_count += 1
                    sub_context_similarity_sum *= self.pcos(self.wvecs.query(sub_word), self.wvecs.query(context_word))
            if context_words_count == 0: context_words_count = 1
            similarities[sub_word] = ((self.pcos(sub_word, target_word) ** context_words_count) * sub_context_similarity_sum) ** (1 / (2 * context_words_count))


        similarities = sorted(similarities.items(), key=itemgetter(1), reverse=True)
        return [similarity[0] for similarity in similarities]

    def substitutes(self, index, sentence, similarity_measure=0):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."

        # if similarity_measure == 5:
        #     average_context = self.wvecs.query(sentence[index])
        #
        #     pruned_context = []
        #     for context_word in sentence[index-5:index+5]:
        #         if context_word is not sentence[index] and context_word not in stop_words and not self.is_punctuation(context_word) and not self.is_num(context_word):
        #             pruned_context.append(context_word)
        #
        #     for context_word in pruned_context:
        #         average_context += self.wvecs.query(context_word) ** (1/len(pruned_context))
        #
        #     return(list(map(lambda k: k[0], self.wvecs.most_similar(average_context, topn=self.topn))))

        if similarity_measure == 4:
            return self.balmult_similarity(sentence[index], self.wvecs.most_similar(sentence[index], topn=self.topn), sentence)
        elif similarity_measure == 3:
            return self.mult_similarity(sentence[index], self.wvecs.most_similar(sentence[index], topn=self.topn), sentence)
        elif similarity_measure == 2:
            return self.baladd_similarity(sentence[index], self.wvecs.most_similar(sentence[index], topn=self.topn), sentence)
        elif similarity_measure == 1:
            return self.add_similarity(sentence[index], self.wvecs.most_similar(sentence[index], topn=self.topn), sentence)
        else:
            return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.retrofit.magnitude'), help="word vectors file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    lexsub = LexSub(opts.wordvecfile, int(opts.topn))
    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub.substitutes(int(fields[0].strip()), fields[1].strip().split(), similarity_measure=0)))
