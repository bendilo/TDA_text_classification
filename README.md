# Text Classification via Topological Data Analysis
This repository is a supplement to my master's thesis, "Text Classification via Topological Data Analysis", written during the spring of 2023 at the Norwegian University of Science and Technology (NTNU).

### Datasets
The folder  ``datasets`` contains datasets with the preprocessed and lemmatized human-written and machine-generated texts that were used in the experiments of the thesis.
* The original texts in  ``essays.csv`` were obtained from the [ChatGPT Generated Text Detection Corpus](https://github.com/rexshijaku/chatgpt-generated-text-detection-corpus).
* The original texts in  ``webtext.csv`` and files with names starting with ``gpt2`` were obtained from the [GPT-2 Output Dataset](https://github.com/openai/gpt-2-output-dataset).

### Source Code
The file ``main.py`` contains all of the Python functions used in the experiments.

### Demonstration
A demonstration of the TDA-based classifiers (on the essay dataset) is provided in the notebook ``demonstration.ipynb``.
