Put your answer files here.

Remember to put your README.username in this directory as well.

#### We have implemented 5 models.
1. `UNIGRAM` is a plain unigram model.

2. `BIGRAM WITH STUPID-BACKOFF` uses stupid backoff which is it backsoff to unigram from bigrams if no bigram is found. It multiplies each probability by `delta=0.4` at each back off step. The function `bigram_prob_stupid_backoff` calculates this probabilty.

3. `BIGRAM WITH JM-SMOOTHING` uses lambda to interpolate the bigram and unigram probabilities such that (lambda)(bigram_prob) * (1-lambda)(unigram_prob). The function `bigram_prob_jm_smoothing` calculates this probability.

4. `TRIGRAM WITH STUPID-BACKOFF` uses stupid backoff from tigram to bigram to unigram. At each step we multiply a additional amount of `0.4`. The function `trigram_prob_stupid_backoff` calculates this probability.

5. `TRIGRAM WITH INTERPOLATION-SMOOTHING` uses three lambda values `lambda_1`, `lambda_2` and `lambda_3` to linearly interpolate the trigram, bigram and unigram probabilities respectively such that `lambda_1 + lambda_2 + lambda_3 = 1`. The function `trigram_prob_interpolation` calculates this probability.


#### We found the best results with the linearly interploated trigram model `TRIGRAM WITH INTERPOLATION-SMOOTHING`. So we are submitting that. We have set the `default.py` to run that model by setting `model_version` by default to it.

#### For the trigrams model we used the `data/train.txt` file to extract the trigram counts. The function `extract_trigrams()` does it. The code is set in such a way that on running `default.py` it checks whether the trigram count file `data/count_3w.txt` exists. If not it extracts the trigrams. The function will first extract the `data/train.txt` from `data/train.txt.bz2` and then extract the trigrams.

#### NOTE: We have added the `count_3w.txt` to `answers/` if the trigram counts fail to be extracted.

#### We have added `optparser.add_option("-m", "--model", dest="model_version", default=4, help="log file for debugging", type=int) to the `default.py` to change the model. It goes from 0 to 4.