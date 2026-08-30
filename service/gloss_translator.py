#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rule-based engine that turns a sequence of RSL gloss tokens (simple root concepts: pronouns,
infinitive verbs, nouns, temporal terms, adjectives) into one fluent, grammatically correct
Romanian sentence.

Grammar applied:
- a leading pronoun (EU/TU) selects person; verbs are conjugated to match (Romanian is
  pro-drop, so eu/tu are omitted from the output). A noun immediately before a verb is treated
  as a third-person subject (e.g. "SEF VREA RAPORT" -> "Seful vrea raport.")
- with no pronoun, first person singular is assumed ("MANCA" -> "Mananc.")
- modal verbs (VREA/PUTEA) followed by another verb build the subjunctive ("vreau sa mananc")
- NU negates the following verb or adjective ("nu mananc", "nu sunt obosit")
- motion verbs + place nouns insert the right preposition ("merg la scoala", "merg acasa")
- question words are fronted and force a '?'; with no verb a copula is inserted ("Unde este mama?")
- temporal words are fronted ("Maine merg la munca.")
- adjectives/feelings agree in gender & number with the subject; for eu/tu/noi/voi the gender
  comes from the selected TTS voice (sentences are first-person speech)
- special verbs: PLACEA takes dative clitics ("imi place cafea"), DUREA takes accusative
  ("ma doare"), ODIHNI is reflexive ("ma odihnesc"), FOAME/SETE are dative states ("mi-e foame")

Unknown tokens fall back to data/phrase_map.csv, then to a plain lowercase rendering.
"""
import csv
import os

PHRASE_MAP_PATH = 'data/phrase_map.csv'

# person index: 0=eu, 1=tu, 2=el/ea, 3=noi, 4=voi (2-4 only reachable via noun-subject/question heuristics)
_PRONOUNS = {'EU': (0, None), 'TU': (1, None)}
_COPULA = ['sunt', 'esti', 'este', 'suntem', 'sunteti']

# indicative present per person + 3rd-person subjunctive (for "sa ..." after modals)
_VERBS = {
    'MANCA': (['mananc', 'mananci', 'mananca', 'mancam', 'mancati'], 'manance'),
    'BEA': (['beau', 'bei', 'bea', 'bem', 'beti'], 'bea'),
    'DORMI': (['dorm', 'dormi', 'doarme', 'dormim', 'dormiti'], 'doarma'),
    'MERGE': (['merg', 'mergi', 'merge', 'mergem', 'mergeti'], 'mearga'),
    'VENI': (['vin', 'vii', 'vine', 'venim', 'veniti'], 'vina'),
    'PLECA': (['plec', 'pleci', 'pleaca', 'plecam', 'plecati'], 'plece'),
    'VREA': (['vreau', 'vrei', 'vrea', 'vrem', 'vreti'], 'vrea'),
    'PUTEA': (['pot', 'poti', 'poate', 'putem', 'puteti'], 'poata'),
    'STI': (['stiu', 'stii', 'stie', 'stim', 'stiti'], 'stie'),
    'INTELEGE': (['inteleg', 'intelegi', 'intelege', 'intelegem', 'intelegeti'], 'inteleaga'),
    'VORBI': (['vorbesc', 'vorbesti', 'vorbeste', 'vorbim', 'vorbiti'], 'vorbeasca'),
    'ASCULTA': (['ascult', 'asculti', 'asculta', 'ascultam', 'ascultati'], 'asculte'),
    'VEDEA': (['vad', 'vezi', 'vede', 'vedem', 'vedeti'], 'vada'),
    'CITI': (['citesc', 'citesti', 'citeste', 'citim', 'cititi'], 'citeasca'),
    'SCRIE': (['scriu', 'scrii', 'scrie', 'scriem', 'scrieti'], 'scrie'),
    'LUCRA': (['lucrez', 'lucrezi', 'lucreaza', 'lucram', 'lucrati'], 'lucreze'),
    'AJUTA': (['ajut', 'ajuti', 'ajuta', 'ajutam', 'ajutati'], 'ajute'),
    'IUBI': (['iubesc', 'iubesti', 'iubeste', 'iubim', 'iubiti'], 'iubeasca'),
    'FACE': (['fac', 'faci', 'face', 'facem', 'faceti'], 'faca'),
    'AVEA': (['am', 'ai', 'are', 'avem', 'aveti'], 'aiba'),
    'LUA': (['iau', 'iei', 'ia', 'luam', 'luati'], 'ia'),
    'CUMPARA': (['cumpar', 'cumperi', 'cumpara', 'cumparam', 'cumparati'], 'cumpere'),
    'PLATI': (['platesc', 'platesti', 'plateste', 'platim', 'platiti'], 'plateasca'),
    'ASTEPTA': (['astept', 'astepti', 'asteapta', 'asteptam', 'asteptati'], 'astepte'),
    'TERMINA': (['termin', 'termini', 'termina', 'terminam', 'terminati'], 'termine'),
    'INCEPE': (['incep', 'incepi', 'incepe', 'incepem', 'incepeti'], 'inceapa'),
    'INVATA': (['invat', 'inveti', 'invata', 'invatam', 'invatati'], 'invete'),
    'ODIHNI': (['odihnesc', 'odihnesti', 'odihneste', 'odihnim', 'odihniti'], 'odihneasca'),
}
_MODALS = {'VREA', 'PUTEA'}
_REFLEXIVE_CLITICS = ['ma', 'te', 'se', 'ne', 'va']  # ODIHNI
_DATIVE_CLITICS = ['imi', 'iti', 'ii', 'ne', 'va']  # PLACEA
_ACCUSATIVE_CLITICS = ['ma', 'te', 'o', 'ne', 'va']  # DUREA (3rd defaults to 'o'/'il' by gender)
_STATE_DATIVE = ['mi-e', 'ti-e', 'ii e', 'ne e', 'va e']  # FOAME / SETE

# (masc_sg, fem_sg, masc_pl, fem_pl)
_ADJECTIVES = {
    'FERICIT': ('fericit', 'fericita', 'fericiti', 'fericite'),
    'TRIST': ('trist', 'trista', 'tristi', 'triste'),
    'OBOSIT': ('obosit', 'obosita', 'obositi', 'obosite'),
    'BOLNAV': ('bolnav', 'bolnava', 'bolnavi', 'bolnave'),
    'SANATOS': ('sanatos', 'sanatoasa', 'sanatosi', 'sanatoase'),
    'MARE': ('mare', 'mare', 'mari', 'mari'),
    'MIC': ('mic', 'mica', 'mici', 'mici'),
    'IMPORTANT': ('important', 'importanta', 'importanti', 'importante'),
    'URGENT': ('urgent', 'urgenta', 'urgenti', 'urgente'),
}

_QUESTION_WORDS = {'CE': 'ce', 'CINE': 'cine', 'UNDE': 'unde', 'CAND': 'cand', 'CUM': 'cum', 'DE_CE': 'de ce'}
_TIME_WORDS = {
    'ACUM': 'acum', 'AZI': 'azi', 'MAINE': 'maine', 'IERI': 'ieri', 'DIMINEATA': 'dimineata',
    'SEARA': 'seara', 'NOAPTE': 'la noapte', 'ORA': 'la ora', 'SAPTAMANA': 'saptamana asta',
    'TARZIU': 'tarziu', 'DEVREME': 'devreme',
}
_COURTESY = {'SALUT': 'salut', 'MULTUMESC': 'multumesc', 'SCUZE': 'scuze', 'DA': 'da', 'BINE': 'bine'}
_STATES = {'FOAME': 'foame', 'SETE': 'sete'}
_MOTION_VERBS = {'MERGE', 'VENI', 'PLECA'}

_NOUNS = {
    'MAMA': 'mama', 'TATA': 'tata', 'FRATE': 'fratele', 'SORA': 'sora', 'COPIL': 'copilul',
    'FAMILIE': 'familia', 'PRIETEN': 'prietenul', 'CASA': 'casa', 'SCOALA': 'scoala',
    'MAGAZIN': 'magazin', 'SPITAL': 'spital', 'MEDIC': 'medicul', 'MEDICAMENT': 'medicamentul',
    'APA': 'apa', 'MANCARE': 'mancare', 'CAFEA': 'cafea', 'PAINE': 'paine', 'BANI': 'bani',
    'TELEFON': 'telefonul', 'MASINA': 'masina', 'TOALETA': 'toaleta', 'STRADA': 'strada',
    'NUME': 'numele', 'TIMP': 'timp',
    'MUNCA': 'munca', 'SEDINTA': 'sedinta', 'PROIECT': 'proiectul', 'COLEG': 'colegul',
    'SEF': 'seful', 'EMAIL': 'emailul', 'RAPORT': 'raportul', 'PAUZA': 'pauza',
    'INTREBARE': 'o intrebare', 'RASPUNS': 'un raspuns', 'PROBLEMA': 'o problema',
}
# bare (article-free) object form used after verbs/prepositions
_NOUNS_BARE = {
    'FRATE': 'frate', 'COPIL': 'copil', 'PRIETEN': 'prieten', 'MEDIC': 'medic',
    'MEDICAMENT': 'medicament', 'TELEFON': 'telefon', 'NUME': 'nume', 'PROIECT': 'proiect',
    'COLEG': 'coleg', 'SEF': 'sef', 'EMAIL': 'email', 'RAPORT': 'raport',
}
_PLACES = {
    'CASA': ('', 'acasa'), 'SCOALA': ('la', 'scoala'), 'MAGAZIN': ('la', 'magazin'),
    'SPITAL': ('la', 'spital'), 'TOALETA': ('la', 'toaleta'), 'STRADA': ('pe', 'strada'),
    'MUNCA': ('la', 'munca'), 'SEDINTA': ('la', 'sedinta'), 'PAUZA': ('in', 'pauza'),
    'PROIECT': ('la', 'proiect'),
}
_PREP_VERBS = _MOTION_VERBS | {'LUCRA'}

_phrase_map_cache = None


def _fallback_phrase_map():
    global _phrase_map_cache
    if _phrase_map_cache is None:
        mapping = {}
        if os.path.exists(PHRASE_MAP_PATH):
            with open(PHRASE_MAP_PATH, encoding='utf-8-sig') as f:
                for row in csv.reader(f):
                    if row and len(row) >= 2 and row[0].strip():
                        mapping[row[0].strip()] = row[1].strip()
        _phrase_map_cache = mapping
    return _phrase_map_cache


def _adjective_form(token, person, gender):
    masc_sg, fem_sg, masc_pl, fem_pl = _ADJECTIVES[token]
    plural = person in (3, 4)
    if plural:
        return fem_pl if gender == 'f' else masc_pl
    return fem_sg if gender == 'f' else masc_sg


def _object_text(token, after_motion_verb):
    if after_motion_verb and token in _PLACES:
        prep, place = _PLACES[token]
        return f'{prep} {place}'.strip()
    if token in _NOUNS_BARE:
        return _NOUNS_BARE[token]
    if token in _NOUNS:
        return _NOUNS[token]
    return None


def _conjugate(token, person, negated):
    forms, _subj3 = _VERBS[token]
    verb = forms[person]
    if token == 'ODIHNI':
        verb = f'{_REFLEXIVE_CLITICS[person]} {verb}'
    return f'nu {verb}' if negated else verb


def _subjunctive(token, person):
    forms, subj3 = _VERBS[token]
    verb = subj3 if person == 2 else forms[person]
    if token == 'ODIHNI':
        return f'sa {_REFLEXIVE_CLITICS[person]} {verb}'
    return f'sa {verb}'


def translate_gloss(tokens, voice_gender='female'):
    """Translates a list of simple gloss tokens into one fluent Romanian sentence.

    `voice_gender` ('male'/'female') supplies gender agreement for first/second-person
    adjectives, matching the selected TTS voice.
    """
    tokens = [t.strip().upper() for t in tokens if t and t.strip().upper() != 'NONE']
    if not tokens:
        return ''

    person = 0
    gender = 'f' if voice_gender == 'female' else 'm'
    subject_word = None
    pronoun_found = False
    for token in tokens:
        if token in _PRONOUNS:
            person, pron_gender = _PRONOUNS[token]
            pronoun_found = True
            if pron_gender:
                gender = pron_gender
                subject_word = 'el' if pron_gender == 'm' else 'ea'
            break
    tokens = [t for t in tokens if t not in _PRONOUNS]

    time_parts = [_TIME_WORDS[t] for t in tokens if t in _TIME_WORDS]
    question = next((_QUESTION_WORDS[t] for t in tokens if t in _QUESTION_WORDS), None)
    tokens = [t for t in tokens if t not in _TIME_WORDS and t not in _QUESTION_WORDS]

    def _is_verbish(t):
        return t in _VERBS or t in _ADJECTIVES or t in ('PLACEA', 'DUREA') or t in _STATES

    explicit_subject = pronoun_found
    # noun followed by a verb = third-person subject: SEF VREA RAPORT -> "seful vrea ..."
    if not explicit_subject and len(tokens) >= 2 and tokens[0] in _NOUNS and _is_verbish(tokens[1]):
        person = 2
        subject_word = _NOUNS[tokens[0]]
        tokens = tokens[1:]
    # questions with verb+noun default to third person: CAND INCEPE SEDINTA -> "cand incepe sedinta?"
    elif not explicit_subject and question and tokens and tokens[0] in _VERBS:
        person = 2 if len(tokens) >= 2 and tokens[1] in _NOUNS else 1

    parts = []
    negate_next = False
    has_verb = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        if token == 'NU' and nxt and (nxt in _VERBS or nxt in _ADJECTIVES or nxt == 'PLACEA'):
            negate_next = True
            i += 1
            continue

        if token == 'PLACEA':
            has_verb = True
            clitic = _DATIVE_CLITICS[person]
            core = f'{clitic} place' if person != 2 else f'{clitic} place'
            if negate_next:
                core = f'nu {core}'
                negate_next = False
            if nxt and nxt in _VERBS:
                parts.append(f'{core} {_subjunctive(nxt, person)}')
                i += 2
                continue
            obj = _object_text(nxt, False) if nxt else None
            if obj:
                parts.append(f'{core} {obj}')
                i += 2
                continue
            parts.append(core)
            i += 1
            continue

        if token == 'DUREA':
            has_verb = True
            clitic = _ACCUSATIVE_CLITICS[person]
            if person == 2 and gender == 'm':
                clitic = 'il'
            core = f'{clitic} doare'
            if negate_next:
                core = f'nu {core}'
                negate_next = False
            obj = _object_text(nxt, False) if nxt else None
            if obj:
                parts.append(f'{core} {obj}')
                i += 2
                continue
            parts.append(core)
            i += 1
            continue

        if token in _STATES:
            has_verb = True
            parts.append(f'{_STATE_DATIVE[person]} {_STATES[token]}')
            i += 1
            continue

        if token in _VERBS:
            has_verb = True
            if token in _MODALS and nxt and (nxt in _VERBS or nxt == 'ODIHNI'):
                modal = _conjugate(token, person, negate_next)
                negate_next = False
                parts.append(f'{modal} {_subjunctive(nxt, person)}')
                i += 2
                # allow an object after the modal chain: VREA MANCA PAINE
                if i < len(tokens):
                    obj = _object_text(tokens[i], nxt in _PREP_VERBS)
                    if obj:
                        parts.append(obj)
                        i += 1
                continue
            verb_text = _conjugate(token, person, negate_next)
            negate_next = False
            obj = _object_text(nxt, token in _PREP_VERBS) if nxt else None
            if obj:
                parts.append(f'{verb_text} {obj}')
                i += 2
                continue
            parts.append(verb_text)
            i += 1
            continue

        if token in _ADJECTIVES:
            adj = _adjective_form(token, person, gender)
            copula = _COPULA[person]
            core = f'nu {copula} {adj}' if negate_next else f'{copula} {adj}'
            negate_next = False
            has_verb = True
            parts.append(core)
            i += 1
            continue

        if token in _COURTESY:
            parts.append(_COURTESY[token])
            i += 1
            continue

        if token == 'NU':
            parts.append('nu')
            i += 1
            continue

        if token in _NOUNS:
            parts.append(_NOUNS[token])
            i += 1
            continue

        fallback = _fallback_phrase_map().get(token) or token.replace('_', ' ').lower()
        parts.append(fallback)
        i += 1

    body = ' '.join(parts)
    if subject_word and body:
        body = f'{subject_word} {body}'

    if question:
        sentence = f'{question} {body}' if has_verb else (f'{question} este {body}' if body else question)
        punct = '?'
    else:
        sentence = body
        punct = '.'

    if time_parts:
        time_text = ', '.join(time_parts)
        sentence = f'{time_text}, {sentence}' if sentence else time_text

    sentence = ' '.join(sentence.split())
    if not sentence:
        return ''
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in '.!?':
        sentence += punct
    return sentence
