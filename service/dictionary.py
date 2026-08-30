#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Loads a Romanian wordlist and finds the closest match to a fingerspelled letter sequence.

Built entirely locally (rapidfuzz, C-accelerated) rather than a cloud spellcheck API, since:
- it must work offline / with zero network latency during a live call
- the spelled-out words shouldn't be sent to any third-party service
- with a 100k+ word dictionary, pure-Python difflib is too slow for real-time use
"""
import os
import re

from rapidfuzz import fuzz, process

DEFAULT_DICTIONARY_PATH = 'data/ro_words.txt'

_DIACRITIC_MAP = str.maketrans({
    'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ş': 's', 'ț': 't', 'ţ': 't',
    'Ă': 'a', 'Â': 'a', 'Î': 'i', 'Ș': 's', 'Ş': 's', 'Ț': 't', 'Ţ': 't',
})


def _compact(text):
    """Lowercases, folds Romanian diacritics to their base letters, and drops non-letters."""
    return re.sub(r'[^a-zA-Z]', '', text.translate(_DIACRITIC_MAP)).lower()


def load_wordlist(path=DEFAULT_DICTIONARY_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def build_index(wordlist):
    """Pre-computes compact keys once so repeated lookups don't redo this work every call."""
    compact_to_word = {}
    for word in wordlist:
        compact_to_word.setdefault(_compact(word), word)
    return compact_to_word


def closest_word(spelled, index, cutoff=60):
    """Returns the closest dictionary word to the spelled letters using a prebuilt index (see build_index)."""
    compact_spelled = _compact(spelled)
    if not compact_spelled or not index:
        return None

    result = process.extractOne(compact_spelled, index.keys(), scorer=fuzz.ratio, score_cutoff=cutoff)
    return index[result[0]] if result else None


def _feminine_candidate(word):
    lower = word.lower()
    if lower.endswith('ă'):
        return None
    if lower.endswith('os'):
        return word[:-2] + 'oasă'
    if lower.endswith('iu'):
        return word[:-2] + 'ie'
    return word + 'ă'


def _masculine_candidate(word):
    lower = word.lower()
    if lower.endswith('oasă'):
        return word[:-4] + 'os'
    if lower.endswith('ie'):
        return word[:-2] + 'iu'
    if lower.endswith('ă'):
        return word[:-1]
    return None


def build_gender_pairs(wordlist):
    """Finds masculine/feminine word pairs already present in the dictionary (e.g. "obosit"/"obosita").

    Only pairs where both forms are real dictionary entries are kept, so this never invents a word
    that isn't in `ro_words.txt` - it just picks whichever real form matches the requested gender.
    """
    by_lower = {}
    for word in wordlist:
        by_lower.setdefault(word.lower(), word)

    pairs = {}
    for word in wordlist:
        candidate = _feminine_candidate(word)
        match = by_lower.get(candidate.lower()) if candidate else None
        if match:
            masc_word, fem_word = word, match
            pairs[masc_word.lower()] = {'masc': masc_word, 'fem': fem_word}
            pairs[fem_word.lower()] = {'masc': masc_word, 'fem': fem_word}
    return pairs


def inflect_for_gender(word, voice_gender, gender_pairs):
    """Returns `word` swapped to its masculine/feminine counterpart if one exists, else `word` unchanged."""
    entry = gender_pairs.get(word.lower())
    if not entry:
        return word
    return entry['fem'] if voice_gender == 'female' else entry['masc']
