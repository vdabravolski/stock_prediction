import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import os
from gensim.models import KeyedVectors
from nltk.corpus import reuters


from sklearn.manifold import TSNE


EMBEDDINGS = "glove" # can be "glove" or "fasttext"
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
GLOVE_DIR = "./glove.6B"
EMBEDDING_DIM = 100

# TSNE config
PERPLEXITY = 50
TSNE_WORDS = 3000
NAME_PRE = "_{0}w_fin_news".format(TSNE_WORDS)
ANNOTATION_LABELS = ['apple', "microsoft", "windows", "ipad", "stock", "market", "ipad", "iphone",
                     "quarter", "nvidia","aapl","msft", "device", "system", "increase", "decrease",
                     "service", "google", "nokia", "intel", "samsung", "dow", "chemicals", "dupont",
                     "jpmorgan", "drop", "forecast", "car", "app", "qualcomm", "investor", "investment",
                     "price", "target", "china", "usa"]

# Get words from Reuters corpus
# WORDS = [word.lower() for word in reuters.words('training/9865')]
# WORDS += [word.lower() for word in reuters.words('training/9940')]
# WORDS += [word.lower() for word in reuters.words('training/9941')]
# WORDS += [word.lower() for word in reuters.words('training/9942')]
# WORDS += [word.lower() for word in reuters.words('training/9943')]
# WORDS += [word.lower() for word in reuters.words('training/9994')]
#WORDS = list(set(WORDS)) # remove duplications



#WORDS = ["lion", "deer", "tiger", "google", "apple", "facebook", "stock", "bond", "etf"]
#WORDS = ["two", "three", "four", "five", "shakespear", "lion", "deer"]
#WORDS = ["goldman", "bloomberg", "blackrock", "bridgewater", "google","facebook", "apple", "wikipedia", "ted"]

TEXT = ["The legal battles between Qualcomm (NASDAQ:QCOM) and Apple (NASDAQ:AAPL) rage on as " \
       "Qualcomm announces plans to file patent infringement complaints. The complaints will " \
       "file with the International Trade Commission and Federal Court. Qualcomm wants the " \
       "Commission to investigate Apple’s “infringing imports” and ultimately ban imports of " \
       "iPhones and other devices containing Qualcomm tech.",
        "A firm part-owned by Siemens (OTCPK:SIEGY) has been hired to help install electricity "
        "turbines in Crimea, a region subject to EU sanctions, sources told Reuters. While Siemens "
        "has denied the allegations, it said that if one of its customers had re-routed any turbines "
        "to Crimea, the company \"will not provide any deliveries or services for installation, "
        "commissioning support, or warranty.\""]


def preprocess_text():
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(TEXT)
    sequences = tokenizer.texts_to_sequences(TEXT)

    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return word_index, data

def process_corpus(file="/Users/vadzimdabravolski/seeking_alpha/output/News_corpus.jl"):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(open(file))
    word_index = tokenizer.word_index

    sorted_words = sorted(word_index, key=word_index.get) # words, sorted by popularity.

    return word_index, sorted_words

def embeddings_visualization(embeddings_index, words, labels):
    target_matrix = []
    cleaned_words = [] # list of words but without elements which don't have embeddings.


    for idx, word in enumerate(words):
        try:
            target_matrix.append(embeddings_index[word])
            cleaned_words.append(word)
        except KeyError:
            print("Keyword {0} with idx {1} is not found in list of embeddings. Ommitting it...".format(word, idx))

    model = TSNE(perplexity=PERPLEXITY, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    reduced_matrix = model.fit_transform(target_matrix)

    fig, ax = plt.subplots()

    max_x = np.amax(reduced_matrix, axis=0)[0]
    max_y = np.amax(reduced_matrix, axis=0)[1]
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))


    ax.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

    for idx, word in enumerate(cleaned_words):
        x = reduced_matrix[idx, 0]
        y = reduced_matrix[idx, 1]
        if word in labels:
            ax.annotate(word, (x, y))

    #plt.show()
    plt.savefig("tsne_charts/TSNE_{0}_PRPL={1}{2}.png".format(EMBEDDINGS,PERPLEXITY, NAME_PRE), dpi=1000)

def retrieve_glove():
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index

def retrieve_fast_test():
    # inspired: https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
    # Creating the model
    en_model = KeyedVectors.load_word2vec_format('fasttext/wiki.en.vec')

    # Getting the tokens
    words = []
    for word in en_model.vocab:
        words.append(word)

    # Printing out number of tokens available
    print("Number of Tokens: {}".format(len(words)))

    # Printing out the dimension of a word vector
    print("Dimension of a word vector: {}".format(
        len(en_model[words[0]])
    ))

    # Print out the vector of a word
    print("Vector components of a word: {}".format(
        en_model[words[0]]
    ))

    # # Pick a word
    # find_similar_to = 'car'
    #
    # # Finding out similar words [default= top 10]
    # for similar_word in en_model.similar_by_word(find_similar_to):
    #     print("Word: {0}, Similarity: {1:.2f}".format(
    #         similar_word[0], similar_word[1]
    #     ))

    return en_model


embeddings_index, sorted_words = process_corpus()

if EMBEDDINGS == "glove":
    embeddings_index = retrieve_glove()
elif EMBEDDINGS == "fasttext":
    embeddings_index = retrieve_fast_test()

embeddings_visualization(embeddings_index, sorted_words[:TSNE_WORDS], ANNOTATION_LABELS)

# print('Found %s word vectors.' % len(embeddings_index))
# print('Found %s word vectors.' % embeddings_index['google'])

# embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector

# print(embedding_matrix[word_index['reuters']])
# print('Found %s word vectors.' % embeddings_index['reuters'])
# print(np.shape(embedding_matrix))
# print(embedding_matrix[0])

