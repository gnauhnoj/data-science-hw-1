import unicodedata
import string
import codecs
import random


def clean_unicode(instr):
    """
    Helper to return unicode string as ascii with special characters removed.
    """
    return unicodedata.normalize('NFKD', instr).encode('ascii', 'ignore')


def clean_ngram(instr):
    """
    Helper to clean ngram unicode strings with punctuation, capital letters, and special
    characters removed.
    """
    # look into another way of doing this
    instr = clean_unicode(instr)
    # return instr.lower().split('_')[0]
    instr = clean_str(instr, True)
    instr = instr.split()
    try:
        instr[0]
    except:
        pass
    else:
        return instr[0]


def clean_str(instr, punc_to_whitespace=False):
    """
    Helper to return string with punctuation and capital letters removed.
    """
    if punc_to_whitespace:
        table = string.maketrans(string.punctuation,
                                 ' '*len(string.punctuation))
        return instr.lower().translate(table)
    return instr.lower().translate(None, string.punctuation)


def get_words(filename):
    """
    Helper to extrct a set of words from the provided filename.
    Assumes each word is the first consecutive group of characters in each line
    """
    words = set([])
    for line in open(filename, 'r'):
        line = clean_str(line, True)  # assumes each line is one word
        line = line.split()
        words.add(line[0])
    return words


def get_years(filename):
    """
    Retrieve a range of years from the provided text file
    Assumes text copied directly from wikipedia in the format: Date1-Date2 (optional) Name of Event
    """
    years = {}
    for line in codecs.open(filename, 'r', 'utf-8'):
        line = clean_unicode(line)
        line = clean_str(line, True)  # assumes each line is one word
        line = line.split()
        date1 = int(line[0])
        try:
            date2 = int(line[1])
        except:
            date2 = date1
            rest_index = 1
        else:
            rest_index = 2
        event_name = " ".join(line[rest_index:len(line)])
        for date in xrange(date1, date2+1):
            try:
                years[date]
            except:
                years[date] = [event_name]
            else:
                years[date].append(event_name)
    return years


def build_map(setify):
    """
    Creates a dictionary that represents the order of a passed set.
    """
    word_map = {}
    listify = list(setify)
    for word in setify:
        word_map[word] = listify.index(word)
    return word_map
