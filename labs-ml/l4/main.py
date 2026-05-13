import numpy as np
from sklearn import preprocessing


def normalize_data(train_data, test_data, type=None):
    if type is None:
        return train_data, test_data
    elif type == "l1":
        train_l1_norms = np.linalg.norm(train_data, ord=1, axis=1, keepdims=True)
        test_l1_norms = np.linalg.norm(test_data, ord=1, axis=1, keepdims=True)
        return train_data / train_l1_norms, test_data / test_l1_norms
    elif type == "l2":
        train_l2_norms = np.linalg.norm(train_data, ord=2, axis=1, keepdims=True)
        test_l2_norms = np.linalg.norm(test_data, ord=2, axis=1, keepdims=True)

        # if we have 0 vector
        train_l2_norms[train_l2_norms == 0] = 1
        test_l2_norms[test_l2_norms == 0] = 1

        return train_data / train_l2_norms, test_data / test_l2_norms
    else:
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)
        scaled_test = scaler.transform(test_data)
        return scaled_train, scaled_test


class BagOfWords:
    def __init__(self) -> None:
        self.vocab = {}
        self.vocab_order = []
        self.next_idx = 1

    def build_vocab(self, data):
        for sent in data:
            for word in sent:
                if word.lower() not in self.vocab:
                    self.vocab[word] = self.next_idx
                    self.next_idx += 1
                    self.vocab_order.append(word.lower())

    def vocab_size(self) -> int:
        return len(self.vocab.keys())


if __name__ == "__main__":
    train_x = np.load("data/training_sentences.npy", allow_pickle=True)
    train_y = np.load("data/training_labels.npy")

    test_x = np.load("data/test_sentences.npy", allow_pickle=True)
    test_y = np.load("data/test_labels.npy")

    print(train_x.shape, train_y.shape)
    print(train_x[0])
    print(train_y[0])

    bag = BagOfWords()
    bag.build_vocab(train_x)

    print(bag.vocab_size())
