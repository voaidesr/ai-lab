from typing import List

import numpy as np
import matplotlib.pyplot as plt


def a() -> np.ndarray:
    images = []
    for i in range(9):
        path = f"./images/car_{i}.npy"
        images.append(np.load(path))
    return np.array(images)

def b(images : np.ndarray) -> int:
    return np.sum(images)

def c(images : np.ndarray) -> List[int]:
    return np.sum(images, axis=(1,2))

def d(image_sum : List[int]) -> int:
    list_with_idx = [(image_sum[i], i) for i in range(len(image_sum))]
    (_, idx) = max(list_with_idx)
    return idx

def e(images: np.ndarray) -> None:
    mean_img = np.mean(images, axis=0)
    plt.imshow(mean_img.astype(np.uint8), cmap="gray")
    plt.show()

def f(images: np.ndarray) -> np.float64:
    return np.std(images).astype(np.float64)

def g(images: np.ndarray) -> np.ndarray:
    normalized_ims = (images - np.mean(images, axis=0)) / f(images)
    return(normalized_ims)

def h(images: np.ndarray) -> np.ndarray:
    return images[:, 200:300, 280:400]

def main():
    images = a()
    print(b(images))
    list_sums = c(images)
    print(list_sums)
    print(d(list_sums))
    e(images)
    print(f(images))
    g(images)

if __name__ == "__main__":
    main()
