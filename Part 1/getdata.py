from google_ngram_downloader import readline_google_store
from utils import get_words, get_years
from get_ngram_data import parse_ngram_file
import matplotlib.pyplot as plt

# Goal:
# X: [for each year {word in words : frequency}]
# Y: [1800 + i for i in xrange(0, 200)] - 1 or -1 / 1 or 0 depending on classifier
# set up a file where all the data gets put into


def build_y(all, positive):
    y_vec = []
    for year in all:
        if year in positive:
            tuple = (year, 1)
        else:
            tuple = (year, 0)
        y_vec.append(tuple)
    return y_vec


def plot_words(word_results):
    for


if __name__ == '__main__':
    all_years = [1800 + i for i in xrange(0, 201)]
    positive_years = get_years('wars.csv')
    word_results = parse_ngram_file('data.csv')
    target_words = get_words('words.csv').intersection(word_results.keys())
    x, y = zip(*word_results['wrangle'])
    plt.scatter(x, y)

    y_vec = build_y(all_years, positive_years)
