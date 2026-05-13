import numpy as np
import matplotlib.pyplot as plt


class KnnClassifier:
    def __init__(self, train_images, train_labels) -> None:
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric="l2"):

        if metric == "l1":
            distances = np.sum(np.abs(self.train_images - test_image), axis=1)
        else:
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))

        knn_indexes = np.argsort(distances)[:num_neighbors]
        knn_labels = self.train_labels[knn_indexes]
        label_counts = np.bincount(knn_labels)

        return np.argmax(label_counts)


def test_Knn(train, train_labels, test, test_labels, metric="l2"):
    ks = [1, 3, 5, 7, 9]
    scores = []

    classifier = KnnClassifier(train, train_labels)
    for k in ks:
        predictions = []
        for img in test:
            prediction = classifier.classify_image(img, k, metric)
            predictions.append(prediction)

        predictions = np.array(predictions)
        acc = np.mean(predictions == test_labels)
        scores.append(acc)

    return scores


if __name__ == "__main__":
    train_images = np.loadtxt("data/train_images.txt")  # incarcam imaginile
    train_labels = np.loadtxt(
        "data/train_labels.txt", "int"
    )  # incarcam etichetele avand
    test_images = np.loadtxt("data/test_images.txt")
    test_labels = np.loadtxt("data/test_labels.txt", "int")

    predictions = []
    classifier = KnnClassifier(train_images, train_labels)

    for img in test_images:
        prediction = classifier.classify_image(img)
        predictions.append(prediction)

    predictions = np.array(predictions)
    acc = np.mean(predictions == test_labels)

    np.savetxt("predictii_3nn_l3_mnist.txt", predictions, fmt="%d")
    print(f"Accuracy for 3nn: {acc}")

    scores_l2 = test_Knn(train_images, train_labels, test_images, test_labels)
    np.savetxt("acuratete_l2.txt", scores_l2, fmt="%.4f")

    scores_l1 = test_Knn(train_images, train_labels, test_images, test_labels, "l1")
    np.savetxt("acuratete_l1.txt", scores_l2, fmt="%.4f")

    plt.plot(scores_l1)
    plt.plot(scores_l2)
    plt.show()
