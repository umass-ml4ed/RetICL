import matplotlib.pyplot as plt

k_values = ["2", "3", "4", "5"]

random = [57.19, 59.894, 60.576, 61.183]
knn = [59.742, 62.092, 61.713, 62.699]
reticl = [65.96, 67.4, 64.37, 65.73]

def main():
    plt.plot(k_values, random, label="GSM8k - Random")
    plt.plot(k_values, knn, label="GSM8k - kNN")
    plt.plot(k_values, reticl, label="GSM8k - RetICL")
    plt.xlabel("In-Context Examples in Prompt")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("k_scaling.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 15, "figure.figsize": (7, 6)})
    main()
