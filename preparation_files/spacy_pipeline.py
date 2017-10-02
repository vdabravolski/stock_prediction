import spacy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
import pandas as pd


DATA_FILE = '/Users/vadzimdabravolski/seeking_alpha/output/News_corpus.jl'
TOKENIZED_DATA_FILE = '/Users/vadzimdabravolski/seeking_alpha/output/News_corpus_tokenized.pickle'
# TSNE config
PERPLEXITY = 50
TSNE_WORDS = 3000 # todo: not used currently
NAME_PRE = "_{0}w_fin_news".format(TSNE_WORDS)
ANNOTATION_LABELS = ['apple', "microsoft", "windows", "ipad", "stock", "market", "ipad", "iphone",
                     "quarter", "nvidia","aapl","msft", "device", "system", "increase", "decrease",
                     "service", "google", "nokia", "intel", "samsung", "dow", "chemicals", "dupont",
                     "jpmorgan", "drop", "forecast", "car", "app", "qualcomm", "investor", "investment",
                     "price", "target", "china", "usa"]

# TODO:
# Following improvements can be done:
# - calculate sentiment of the sentences
# - calculate averaging sentence embedding vector


def get_tokenized_text(save=False, load_saved=True):

    data_tokenized = {}

    # if we choose to use pre-saved data
    if load_saved:
        with open(TOKENIZED_DATA_FILE, 'rb') as f:
            data_tokenized = pickle.load(f)
        return data_tokenized


    # otherwise, we will open file, tokenize it and return result
    with open(DATA_FILE,'r') as f:
        data = f.read()
    npl = spacy.load('en')
    doc = npl(data)

    for i, sent in enumerate(doc.sents):
        for token in sent:
            if not token.is_punct:
                if not token.is_space:
                    if not token.like_url:
                        if not token.like_num:
                            data_tokenized[i] = (token.lemma_, token.vector)
                            print(token.lemma_, token.vector)
        print("Sentence #{0} processed".format(i))

    if save:
        with open(TOKENIZED_DATA_FILE, 'wb') as f:
            pickle.dump(data_tokenized, f)

    return data_tokenized


def embeddings_visualization(text, labels):
    vectors = []
    words = []

    for _, item in text.items():
        words.append(item[0]) # append word itself
        vectors.append(item[1]) # append word vector


    print("Training TSNE representations")
    model = TSNE(perplexity=PERPLEXITY, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    reduced_matrix = model.fit_transform(vectors)

    fig, ax = plt.subplots()

    max_x = np.amax(reduced_matrix, axis=0)[0]
    max_y = np.amax(reduced_matrix, axis=0)[1]
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))


    ax.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

    for idx, word in enumerate(words):
        x = reduced_matrix[idx, 0]
        y = reduced_matrix[idx, 1]
        if word in labels:
            ax.annotate(word, (x, y))

    plt.show()
    plt.savefig("tsne_charts/TSNE_{0}_PRPL={1}{2}.png".format('glove',PERPLEXITY, 'tokenized'), dpi=1000)


if __name__ == '__main__':
    text = get_tokenized_text()
    embeddings_visualization(text, labels=ANNOTATION_LABELS)


