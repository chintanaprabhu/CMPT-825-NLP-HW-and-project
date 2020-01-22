#!/usr/bin/env bash

echo "Modifying word vectors."
python3 modifyWordVec.py -l data/lexicons/wordnet-synonyms.txt
echo "Finished. Creating .magnitude file now"
python3 -m pymagnitude.converter -i data/glove.6B.100d.retrofit.txt -o data/glove.6B.100d.retrofit.magnitude
