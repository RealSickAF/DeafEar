#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Glues the gesture recognizer to Romanian speech synthesis and virtual audio output."""
import csv
import os
import queue
import threading

from audio.output_router import play
from engine.recognizer import GestureRecognizer
from service.dictionary import build_gender_pairs, build_index, closest_word, inflect_for_gender, load_wordlist
from service.fingerspell_buffer import DEFAULT_CONFIRM_DELAY, DEFAULT_FINALIZE_TIMEOUT, FingerspellBuffer
from service.gloss_buffer import DEFAULT_FINALIZE_TIMEOUT as DEFAULT_SENTENCE_PAUSE, GlossBuffer
from service.gloss_translator import translate_gloss
from tts.speech_engine import synthesize

PHRASE_MAP_PATH = 'data/phrase_map.csv'


def load_phrase_map():
    mapping = {}
    if os.path.exists(PHRASE_MAP_PATH):
        with open(PHRASE_MAP_PATH, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if row and len(row) >= 2 and row[0].strip():
                    mapping[row[0].strip()] = row[1].strip()
    return mapping


def label_to_phrase(label, phrase_map):
    if label in phrase_map:
        return phrase_map[label]
    return label.replace('_', ' ').lower()


class TranslatorService:
    """Runs recognition in the background and speaks each newly detected gesture.

    In 'words' mode, a detected sign is spoken immediately. In 'letters' mode, letters are
    accumulated by a FingerspellBuffer and only spoken once resolved into a word.
    """

    def __init__(
        self,
        camera_index=0,
        voice_gender='female',
        output_device_index=None,
        mode='words',
        on_status=None,
        letter_hold_delay=DEFAULT_CONFIRM_DELAY,
        word_pause=DEFAULT_FINALIZE_TIMEOUT,
        sentence_pause=DEFAULT_SENTENCE_PAUSE,
    ):
        self.voice_gender = voice_gender
        self.output_device_index = output_device_index
        self.on_status = on_status
        wordlist = load_wordlist()
        self._word_index = build_index(wordlist)
        self._gender_pairs = build_gender_pairs(wordlist)

        self._speech_queue = queue.Queue()
        self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._speech_thread.start()

        self._fingerspell = FingerspellBuffer(
            resolve_word_fn=self._resolve_word,
            on_word=self._handle_word_resolved,
            on_letter=self._handle_letter_progress,
            confirm_delay=letter_hold_delay,
            finalize_timeout=word_pause,
        )

        self._gloss_buffer = GlossBuffer(
            translate_fn=lambda tokens: translate_gloss(tokens, voice_gender=self.voice_gender),
            on_sentence=self._handle_sentence_resolved,
            on_progress=self._handle_gloss_progress,
            finalize_timeout=sentence_pause,
        )

        self.recognizer = GestureRecognizer(
            camera_index=camera_index,
            mode=mode,
            on_gesture=self._handle_gesture,
        )

    def set_mode(self, mode):
        if mode != self.recognizer.mode:
            self._fingerspell.reset()
            self._gloss_buffer.reset()
        self.recognizer.set_mode(mode)

    def _resolve_word(self, spelled):
        word = closest_word(spelled, self._word_index) or spelled.lower()
        return inflect_for_gender(word, self.voice_gender, self._gender_pairs)

    def _handle_letter_progress(self, spelled_so_far):
        if self.on_status:
            self.on_status(f'Spelling: {spelled_so_far}')

    def _handle_word_resolved(self, spelled, word):
        if self.on_status:
            self.on_status(f'Word: {spelled} -> "{word}"')
        self._speech_queue.put(word)

    def _handle_gesture(self, label):
        if self.recognizer.mode == 'letters':
            self._fingerspell.feed(label)
            return
        self._gloss_buffer.feed(label)

    def _handle_gloss_progress(self, tokens):
        if self.on_status:
            self.on_status(f'Gloss: {" ".join(tokens)}')

    def _handle_sentence_resolved(self, tokens, sentence):
        if self.on_status:
            self.on_status(f'Gloss: {" ".join(tokens)} -> "{sentence}"')
        self._speech_queue.put(sentence)

    def _speech_worker(self):
        while True:
            phrase = self._speech_queue.get()
            try:
                pcm, rate = synthesize(phrase, gender=self.voice_gender)
                play(pcm, rate, device_index=self.output_device_index)
            except Exception as exc:
                if self.on_status:
                    self.on_status(f'Speech error: {exc}')

    def start(self):
        self.recognizer.start()

    def stop(self):
        self.recognizer.stop()

    @property
    def is_running(self):
        return self.recognizer.is_running
