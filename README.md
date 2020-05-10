# KeyPhraseExtraction

Automatic keyword extraction is an important research topic. Keywords serve as a dense summary for a document, lead to improved information retrieval, or be the entrance to a document collection. The aim of this project is to find a small set of terms that describes a specific document, independently of the domain it belongs to. 

### Models
We examine five different models: n-grams (unigram, bigram, trigram), POSTagger-based-ngrams (terms matching any of a set of part-of-speech (POS) tag sequences), and RAKE (Rapid Automatic Keyword Extraction). 

For the models using ngrams, we use three different features: term frequency, collection frequency and relative position of the first occurrence to find top keywords. 

### Installation & Usage
To run our model, download every file in the repo into one directory and run KeyPhraseExtraction.java file. Unzip the TRAINING.zip to get training data. 

Files ending with ".abstr" are documents used to train our models, while ".contr" contains the controlled manually assigned keywords, separated with semicolon and ".uncontr" contains the uncontrolled manually assigned keywords, separated with semicolon. 

For POSTagger model, please download the basic English Stanford Tagger (http://nlp.stanford.edu/software/tagger.shtml) and add ".jar" files to your project build path or class path. 

### Testing & Evaluation
To evaluate the performance, we compute the precision score of the top 5 keywords generated for each document in the Training Dataset and compare among 5 models. 

The result is presented in the report.txt file. The standard used to calculate precision score of each ".abstr" document is the manually-picked keywords stored in the corresponding ".uncontr" files.
