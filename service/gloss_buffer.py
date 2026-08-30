#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accumulates recognized gesture gloss tokens and resolves them into one fluent sentence
after a pause, mirroring FingerspellBuffer's debounce-then-finalize approach but for
whole-word/phrase gestures instead of letters.
"""
import threading

DEFAULT_FINALIZE_TIMEOUT = 2.5


class GlossBuffer:
    def __init__(self, translate_fn, on_sentence=None, on_progress=None, finalize_timeout=DEFAULT_FINALIZE_TIMEOUT):
        self.translate_fn = translate_fn
        self.on_sentence = on_sentence
        self.on_progress = on_progress
        self.finalize_timeout = finalize_timeout

        self._lock = threading.Lock()
        self._tokens = []
        self._finalize_timer = None

    def feed(self, label):
        """Call with each newly detected stable gesture label while in words mode."""
        with self._lock:
            self._tokens.append(label)
            tokens_so_far = list(self._tokens)
            if self._finalize_timer:
                self._finalize_timer.cancel()
            self._finalize_timer = threading.Timer(self.finalize_timeout, self._finalize)
            self._finalize_timer.daemon = True
            self._finalize_timer.start()

        if self.on_progress:
            self.on_progress(tokens_so_far)

    def _finalize(self):
        with self._lock:
            tokens = self._tokens
            self._tokens = []
            self._finalize_timer = None

        if not tokens:
            return

        sentence = self.translate_fn(tokens)
        if self.on_sentence:
            self.on_sentence(tokens, sentence)

    def reset(self):
        """Discards any in-progress buffer, e.g. when leaving words mode."""
        with self._lock:
            if self._finalize_timer:
                self._finalize_timer.cancel()
            self._finalize_timer = None
            self._tokens = []
