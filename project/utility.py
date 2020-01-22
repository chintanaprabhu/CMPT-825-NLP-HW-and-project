import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# downloading stopwords and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# apostrophe words dict
APPO = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "didn't": "did not"
}

stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

def clean(comment):
    # converting to lowercase
    comment = comment.lower()

    # replacing \n with space
    comment = re.sub("\\n", " ", comment)

    # replacing \t with space
    comment = re.sub("\\t", " ", comment)

    # remove leaky elements like ip and usernames
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment) # for ip
    comment = re.sub("\[\[.*\]", "", comment) # for usernames

    # Split the sentences into words
    words = tokenizer.tokenize(comment)

    # replacing apostrophe contiaing words to their raw form
    words = [APPO[word] if word in APPO else word for word in words]

    # splitting the words
    words = [word.split() for word in words]
    words = [item for sublist in words for item in sublist]

    # lemmatizing words
    words = [lemmatizer.lemmatize(word, "v") for word in words]

    # removing stop words
    words = [w for w in words if not w in stopwords]

    # joining the words
    clean_sent = " ".join(words)

    # removing non alphanum and digit characters
    clean_sent=re.sub("\W+"," ",clean_sent)

    return (clean_sent)
