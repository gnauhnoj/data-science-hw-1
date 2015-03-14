from google_ngram_downloader import readline_google_store
from get_ngram_data import parse_ngram_file, transform_to_year
from helpers import get_words, get_years, build_map
import matplotlib.pyplot as plt
import numpy as np
import os.path
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, svm, neighbors, qda, metrics, cross_validation
from sklearn.pipeline import Pipeline


def build_outcome(all_things, positive):
    y_vec = []
    for year in all_things:
        if year in positive:
            tup = (year, 1)
        else:
            tup = (year, 0)
        y_vec.append(tup)
    return y_vec


def reservoir_sample(input, N):
    sample = []
    for i, line in enumerate(input, start=1):
        if i <= N:
            sample.append(line)
        elif i > N and random.random() < N/i:
            replace = random.randint(0, len(sample)-1)
            sample[replace] = line
    return sample


def plot_words(word_results):
    for word in word_results:
        x, y = zip(*word_results[word])
        y = standardize_counts(y)
        plt.plot(x, y)
    plt.show()
    plt.close()
# need to add labels... somehow
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     left='off',
#     right='off',
#     labelbottom='off',
#     labelleft='off')   # labels along the bottom edge are off
# plt.ylabel('P(TP) - True Positive Rate')
# plt.xlabel('P(FP) - False Positive Rate')
# plt.title('Problem 2a-2d: ROC Curve Predictions')
# plt.legend(scatterpoints=1, loc='lower center')
# plt.savefig('Random.png', dpi=100)


def balance_pool(pool):
    newpool = {}
    neg = []
    pos = []
    for year in pool:
        if pool[year][-1] == 0:
            neg.append((year, pool[year]))
        else:
            pos.append((year, pool[year]))
    minlen = min(len(pos), len(neg))
    for elem in reservoir_sample(neg, minlen):
        newpool[elem[0]] = elem[1]
    for elem in reservoir_sample(pos, minlen):
        newpool[elem[0]] = elem[1]
    return newpool


def create_data_pool(outcome_vec, word_map, year_data):
    # for each year in outcome_vec
    # generate a vector of length len(words)
    pool = {}
    for tup in outcome_vec:
        year, outcome = tup
        pool[year] = [0] * (len(word_map.keys()) + 1)
        pool[year].append(outcome)
        for word_tup in year_data[year]:
            word, count = word_tup
            pool[year][word_map[word]] = count
    return pool


def create_train_test(pool, trainfile, testfile):
    points = pool.values()
    random.shuffle(points)
    ftrain = open(trainfile, 'w')
    ftest = open(testfile, 'w')
    ind = int(floor(len(points) * 0.8))
    train = points[:ind]
    test = points[ind:]
    for point in train:
        ftrain.write('||'.join(str(x) for x in point))
        ftrain.write('\n')
    for point in test:
        ftest.write('||'.join(str(x) for x in point))
        ftest.write('\n')


def parse_train_test(filename):
    for line in open(filename, 'r'):
        fields = line.split("||")
        try:
            yield [float(x) for x in fields]
        except:
            print fields


def BuildXY(filename):
    x, y = [], []
    listify = list(parse_train_test(filename))
    for year in listify:
        y.append(year[-1])
        x.append(year[:-1])
    return [x, y]


if __name__ == '__main__':
    if not os.path.isfile('train.txt') or not os.path.isfile('test.txt'):
        word_results = parse_ngram_file('data.csv')
        target_words = get_words('words.csv').intersection(word_results.keys())
        year_data = transform_to_year(word_results)
        positive_years = get_years('wars.csv')
        outcome_vec = build_outcome(year_data.keys(), positive_years)
        word_map = build_map(target_words)
        pool = create_data_pool(outcome_vec, word_map, year_data)
        pool = balance_pool(pool)
        create_train_test(pool, 'train.txt', 'test.txt')
    train = BuildXY('train.txt')
    clf = Pipeline([('Scaler', StandardScaler()),
                    ('Log-Reg', linear_model.LogisticRegression(penalty='l2', dual=True))])
                    # ('kNN', neighbors.KNeighborsClassifier())])
                    # ('SVC-linear', svm.SVC(kernel='linear', C=1))])
                    # ('SVC-rbf', svm.SVC(kernel='rbf'))])
    cv = cross_validation.KFold(len(train[0]), n_folds=5, shuffle=True)
    scores = cross_validation.cross_val_score(clf, train[0], train[1], cv=cv)
    print np.average(scores)
    test = BuildXY('test.txt')
    clf = clf.fit(train[0], train[1])
    predicted = clf.predict(test[0])
    print metrics.accuracy_score(test[1], predicted)
    print metrics.confusion_matrix(test[1], predicted)
    print metrics.classification_report(test[1], predicted)
