""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols, _pad
import os
from dp.phonemizer import Phonemizer
phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def do_ipa(text):
    text = text.split()
    new_text = []
    for t in text:
        if t.startswith('*'):
            t = t.replace("*", "")
            new_text.append(t)
        else:    
            t = phoneme_text(t)
            new_text.append(t)
    
    return ' '.join(new_text)


def phoneme_text(text, lang='en_us'):
    text = phonemizer(text, lang)
    return text.strip()

def text_to_sequence(text, symbols, cleaner_names):
    sequence = []
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    clean_text = _clean_text(text, cleaner_names)
    print(clean_text)
    clean_text = phoneme_text(clean_text)
    print(clean_text)
    print(f" length:{len(clean_text)}")
    for symbol in clean_text:
        symbol_id = symbol_to_id[symbol]
        sequence += [symbol_id]
    sequence = sequence + [_symbol_to_id[_pad]]
    print(f" length:{len(sequence)}")
    return sequence


def cleaned_text_to_sequence(cleaned_text, symbols):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    '''
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    sequence = [symbol_to_id[symbol] for symbol in cleaned_text if symbol in symbol_to_id.keys()]
    # sequence = sequence + [_symbol_to_id[_pad]]
    return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
