#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Accumulates fingerspelled letters and resolves them into a word after a pause.

Each letter is only committed after it stays stable for `confirm_delay` seconds (debounce),
and the accumulated letters are resolved into a word after `finalize_timeout` seconds of silence.
"""
import threading

DEFAULT_CONFIRM_DELAY = 0.5
DEFAULT_FINALIZE_TIMEOUT = 2.5


class FingerspellBuffer:
    def __init__(
        self,
        resolve_word_fn,
        on_word=None,
        on_letter=None,
        confirm_delay=DEFAULT_CONFIRM_DELAY,
        finalize_timeout=DEFAULT_FINALIZE_TIMEOUT,
    ):
        self.resolve_word_fn = resolve_word_fn
        self.on_word = on_word
        self.on_letter = on_letter
        self.confirm_delay = confirm_delay
        self.finalize_timeout = finalize_timeout

        self._lock = threading.Lock()
        self._letters = []
        self._pending_letter = None
        self._confirm_timer = None
        self._finalize_timer = None

    def feed(self, label):
        """Call with each newly detected stable letter while in letters mode."""
        with self._lock:
            if self._confirm_timer:
                self._confirm_timer.cancel()
            self._pending_letter = label
            self._confirm_timer = threading.Timer(self.confirm_delay, self._confirm, args=(label,))
            self._confirm_timer.daemon = True
            self._confirm_timer.start()

    def _confirm(self, label):
        with self._lock:
            if self._pending_letter != label:
                return  # a different letter arrived before this one was confirmed
            self._letters.append(label)
            spelled_so_far = ''.join(self._letters)
            self._pending_letter = None

            if self._finalize_timer:
                self._finalize_timer.cancel()
            self._finalize_timer = threading.Timer(self.finalize_timeout, self._finalize)
            self._finalize_timer.daemon = True
            self._finalize_timer.start()

        if self.on_letter:
            self.on_letter(spelled_so_far)

    def _finalize(self):
        with self._lock:
            letters = self._letters
            self._letters = []
            self._finalize_timer = None

        if not letters:
            return

        spelled = ''.join(letters)
        word = self.resolve_word_fn(spelled)
        if self.on_word:
            self.on_word(spelled, word)

    def reset(self):
        """Discards any in-progress buffer, e.g. when leaving letters mode."""
        with self._lock:
            if self._confirm_timer:
                self._confirm_timer.cancel()
            if self._finalize_timer:
                self._finalize_timer.cancel()
            self._confirm_timer = None
            self._finalize_timer = None
            self._letters = []
            self._pending_letter = None
