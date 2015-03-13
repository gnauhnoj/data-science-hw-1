from google_ngram_downloader import readline_google_store
from utils import clean_ngram, get_words


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
                store[cleaned][year] = count
    for key in store.keys():
        yield (key, store[key])


def yield_file(all_years, target):
    datastore = readline_google_store(ngram_len=1)
    for file in datastore:
        print file[0]
        obj = get_data_file(file, all_years, target)
        yield (file[0], obj)


# THIS TAKES A REALLY LONG TIME
def write_ngrams(all_years, target, filename):
    f = open(filename, 'w')
    for file in yield_file(all_years, target):
        for word in file[1]:
            line = []
            for year in word[1]:
                kv = ':'.join([str(year), str(word[1][year])])
                line.append(kv)
            line = ",".join(line)
            print "word", word[0]
            print "line", line
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


if __name__ == '__main__':
    all_years = [1800 + i for i in xrange(0, 201)]
    target_words = get_words('words.csv')
    # write_ngrams(all_years, target_words, 'data.csv')
