from get_ngram_data import parse_ngram_file, transform_to_year
from helpers import get_words, get_years, build_map
import matplotlib.pyplot as plt
import numpy as np
import os.path
import random
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, svm, neighbors, qda, metrics, cross_validation
from sklearn.pipeline import Pipeline


def build_outcome(all_things, positive):
    """
    Build an outcome vector given a whole set of years and a set of years that represent the positive cross_val_score
    Output format [(year, 1 OR 0)]
    """
    y_vec = []
    for year in all_things:
        if year in positive:
            tup = (year, 1)
        else:
            tup = (year, 0)
        y_vec.append(tup)
    return y_vec


def reservoir_sample(input, N):
    """
    Reservoir sampling given an iterable input and N for number of items to be sampled
    """
    sample = []
    for i, line in enumerate(input, start=1):
        if i <= N:
            sample.append(line)
        elif i > N and random.random() < N/i:
            replace = random.randint(0, len(sample)-1)
            sample[replace] = line
    return sample


def balance_pool(pool):
    """
    Given a pool of year-key formatted unbalanced outcome data, return a balanced set where positive outcomes have equal number of occurances to negative outcomes
    The larger set is randomly sampled to reduce its size by using reservoir_sample
    """
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


def plot_words(word_results):
    """
    Given a pool of word-key format data, plot the number of occurances for all of the words in the word_result dictionary
    """
    for word in word_results:
        x, y = zip(*word_results[word])
        plt.plot(x, y)
    # need to add labels... somehow
    plt.ylabel('1 Gram Frequency (number of occurances)')
    plt.xlabel('Year (1800 - 2000)')
    plt.title('Part 1: Selected 1 Gram Frequencies')
    plt.savefig('Words.png', dpi=100)
    plt.show()
    plt.close()


def create_data_pool(outcome_vec, word_map, year_data):
    """
    Create a data pool from an outcome vector, ordered word map ({word: relative index}), year-key data
    Finds year matches between outcome vector and year-key data, builds a data pool for each year
    Each entry in the datapool is: {year: [...ordered vector of word counts..., outcome value]}
    The output corresponds to each point in our final dataset
    """
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
    """
    Split the data pool created in create_data_pool randomly into a 80/20 split between training data and testing data
    Shuffles all the years and randomly splits 80/20 between training and test
    Should only be ran once to randomly split train/test data as it will return different results between runs
    """
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
    """
    Parses the train/test files created by create_train_test
    Returns the data in a generator form
    """
    for line in open(filename, 'r'):
        fields = line.split("||")
        try:
            yield [float(x) for x in fields]
        except:
            print fields


def BuildXY(filename):
    """
    Creates X and Y vectors from test/train files
    """
    x, y = [], []
    listify = list(parse_train_test(filename))
    for year in listify:
        y.append(year[-1])
        x.append(year[:-1])
    return [x, y]


if __name__ == '__main__':
    # if train and test don't exist, create a data pool (in this case ignoring data from 1700 - 1800)
    # write train and test files
    # assumes data.csv exists
    if not os.path.isfile('train.txt') or not os.path.isfile('test.txt'):
        remove = [1700 + i for i in xrange(0, 100)]
        word_results = parse_ngram_file('data.csv', remove)
        target_words = get_words('words.csv').intersection(word_results.keys())
        year_data = transform_to_year(word_results)
        positive_years = get_years('wars.csv')
        outcome_vec = build_outcome(year_data.keys(), positive_years)
        word_map = build_map(target_words)
        pool = create_data_pool(outcome_vec, word_map, year_data)
        # balance the pool not used in the final version -- reasons discussed in report
        # pool = balance_pool(pool)
        create_train_test(pool, 'train.txt', 'test.txt')
        plot_words(word_results)
    # read train/test files
    train = BuildXY('train.txt')
    test = BuildXY('test.txt')
    # builds a sklearn pipeline of different classifiers (manually modified to choose best one)
    clf = Pipeline([('Scaler', StandardScaler()),
                    # ('Log-Reg', linear_model.LogisticRegression(penalty='l2', dual=True))])
                    # ('Log-Reg', linear_model.LogisticRegression(penalty='l2', dual=False))])
                    # ('kNN', neighbors.KNeighborsClassifier())]) # default k is 5
                    # ('kNN', neighbors.KNeighborsClassifier(n_neighbors=3))]) # default k is 5
                    ('SVC-linear', svm.SVC(kernel='linear'))])
                    # ('SVC-rbf', svm.SVC(kernel='rbf'))])
    # performs kfold cross validation on our selected model, n_folds = 4 (so each validation set is 20% of data, shuffle prior to performing folds to ensure a random validation set (since consecutive years can have same result)
    cv = cross_validation.KFold(len(train[0]), n_folds=4, shuffle=True)
    scores = cross_validation.cross_val_score(clf, train[0], train[1], cv=cv)
    print scores, np.average(scores)
    # peforms test on selected model
    clf = clf.fit(train[0], train[1])
    predicted = clf.predict(test[0])
    print metrics.accuracy_score(test[1], predicted)
