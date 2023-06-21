import matplotlib.pyplot as plt

ratios = ["0.1%", "1%", "10%", "100%"]

tabmwp = (
    ["20", "200", "2000", "all"],
    [79.5, 84.7, 85.5, 88.3],
    [77.2, 84.4, 85.6, 88.2]
)
gsm8k = (
    ["6", "60", "600", "all"],
    [60.35, 62.24, 62.4, 66.111],
    [58.15, 61.107, 57.468, 59.742]
)

def proportional(vals, upper):
    return [val / upper for val in vals]

def main_abs():
    plt.plot(ratios, tabmwp[1], label="TabMWP - RetICL")
    plt.plot(ratios, tabmwp[2], label="TabMWP - kNN")
    plt.plot(ratios, gsm8k[1], label="GSM8k - RetICL")
    plt.plot(ratios, gsm8k[2], label="GSM8k - kNN")
    plt.xlabel("Percentage of Available Example Candidates")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("corpus_size_abs.png", dpi=300)
    plt.show()

def main_rel():
    plt.plot(ratios, proportional(tabmwp[1], tabmwp[1][-1]), label="TabMWP - RetICL", linewidth=3.0)
    plt.plot(ratios, proportional(tabmwp[2], tabmwp[1][-1]), label="TabMWP - kNN", linewidth=3.0)
    plt.plot(ratios, proportional(gsm8k[1], gsm8k[1][-1]), label="GSM8k - RetICL", linewidth=3.0)
    plt.plot(ratios, proportional(gsm8k[2], gsm8k[1][-1]), label="GSM8k - kNN", linewidth=3.0)
    plt.xlabel("Percentage of Available Example Candidates")
    plt.ylabel("Relative Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("corpus_size_rel.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 15, "figure.figsize": (7, 6)})
    main_rel()
