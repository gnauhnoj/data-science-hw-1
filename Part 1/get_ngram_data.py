from google_ngram_downloader import readline_google_store
from helpers import clean_ngram, get_words
import os.path


def get_data_file(file, all_years, target):
    # store is (string : {year: count})
    store = {}
    for single in file[2]:
        cleaned = clean_ngram(single.ngram)
        year = single.year
        count = single.match_count
        if cleaned in target and year in all_years:
            try:
                store[cleaned]
            except:
                store[cleaned] = {year: count}
            else:
                try:
                    store[cleaned][year]
                except:
                    store[cleaned][year] = count
                else:
                    store[cleaned][year] += count
    for key in store.keys():
        yield (key, store[key])


def yield_file(all_years, target, ignore):
    datastore = readline_google_store(ngram_len=1)
    for file in datastore:
        print file[0]
        if file[0] not in ignore:
            obj = get_data_file(file, all_years, target)
            yield (file[0], obj)


# THIS TAKES A REALLY LONG TIME
def write_ngrams(all_years, target, filename, ignore):
    for file in yield_file(all_years, target, ignore):
        f = open(filename, 'a')
        for word in file[1]:
            line = []
            for year in word[1]:
                kv = ':'.join([str(year), str(word[1][year])])
                line.append(kv)
            line = ",".join(line)
            f.write('||'.join([word[0], line]))
            f.write('\n')


def parse_ngram_file(filename):
    f = open(filename, 'r')
    word_results = {}
    for line in f:
        word, date_line = line.split('||')
        dates = date_line.split(',')
        vec = []
        for date in dates:
            date_year, date_count = date.split(':')
            vec.append((int(date_year), int(date_count)))
        word_results[word] = vec
    return word_results


def transform_to_year(word_results):
    store = {}
    for word in word_results:
        for tup in word_results[word]:
            year, count = tup
            try:
                store[year]
            except:
                store[year] = [(word, count)]
            else:
                store[year].append((word, count))
    return store


if __name__ == '__main__':
    all_years = [1700 + i for i in xrange(0, 201)]
    target_words = get_words('words.csv')
    if not os.path.isfile('data.csv'):
        ignore = ['googlebooks-eng-all-1gram-20120701-0.gz',
                  'googlebooks-eng-all-1gram-20120701-1.gz',
                  'googlebooks-eng-all-1gram-20120701-2.gz',
                  'googlebooks-eng-all-1gram-20120701-3.gz',
                  'googlebooks-eng-all-1gram-20120701-4.gz',
                  'googlebooks-eng-all-1gram-20120701-5.gz',
                  'googlebooks-eng-all-1gram-20120701-6.gz',
                  'googlebooks-eng-all-1gram-20120701-7.gz',
                  'googlebooks-eng-all-1gram-20120701-8.gz',
                  'googlebooks-eng-all-1gram-20120701-9.gz',
                  'googlebooks-eng-all-1gram-20120701-other.gz',
                  'googlebooks-eng-all-1gram-20120701-pos.gz',
                  'googlebooks-eng-all-1gram-20120701-punctuation.gz']
        write_ngrams(all_years, target_words, 'data.csv', ignore)
