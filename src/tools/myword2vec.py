import numpy as np
from numpy import fromstring, float32, dtype
from numpy import linalg as LA


class Word2Vec():
    def unitvec(self, vec):
        return (1.0 / LA.norm(vec, ord=2)) * vec

    def __init__(self, path):
        with open(path, 'r') as input:
            header = input.readline()
            self.vocab_size, self.vector_size = map(int, header.split())
            self.vocab = dict()
            self.vocab_reverse = dict()
            self.vectors = np.empty((self.vocab_size, self.vector_size), dtype=np.float)
            for i in xrange(self.vocab_size):
                word = []
                while True:
                    ch = input.read(1)
                    # print ch,
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    word.append(ch)
                self.vocab[i] = word
                self.vocab_reverse[word] = i
                vector = fromstring(input.read(self.vector_size * dtype(float32).itemsize), dtype=float32)
                self.vectors[i] = self.unitvec(vector)
                input.read(1)

    def cosine(self, word1, word2):
        word1 = word1.replace(' ', '_')
        word2 = word2.replace(' ', '_')
        if word1 not in self.vocab_reverse or word2 not in self.vocab_reverse:
            return -1
        return np.dot(self.vectors[self.vocab_reverse[word1]], self.vectors[self.vocab_reverse[word2]])

    def get_vector(self, word):
        word = word.replace(' ', '_')
        if word in self.vocab_reverse:
            return self.vectors[self.vocab_reverse[word]], True
        else:
            return np.zeros(self.vector_size), False
