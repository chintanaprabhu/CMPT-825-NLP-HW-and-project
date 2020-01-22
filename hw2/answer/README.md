Put your answer files here.

Remember to put your README.username in this directory as well.

Make sure the `glove.6B.100d.retrofit.magnitude` file exists in the `/data` folder

To create the retrofitted .magnitude file, run this command:

`sh run.sh`

This script reads the original word vector file `glove.6B.100d.magnitude` from the given CSIL path. 

It will take approximatelty 15 minutes to complete the process. The script runs `modifyWordVec.py` file that generates the retrofiited word vectors in a text file. It is by default reading the `wordnet-synonyms.txt` for reading the lexicon to create ontology graph. The final step generates '.magnitude' file from the generated '.txt' file. 

