import re
from unidecode import unidecode
from .numbers import normalize_numbers
from num2words import num2words


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

def convert_to_ascii(text):
    return unidecode(text)

def lowercase(text):
    return text.lower()

def expand_numbers(text):
    return normalize_numbers(text)

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text
    
def ordinal_dates(text):
    texts = re.findall("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\.)?(\w*)?(\.)?(\s*\d{0,2}\s*),(\s*\d{4})", text)
    for result in texts:
        rs = result[0] + result[2] + result[4]
        word_date = num2words(result[4], to='ordinal')
        rss = result[0] + result[2] + " " + word_date
        text = text.replace(rs, rss)
    return text

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = ordinal_dates(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text