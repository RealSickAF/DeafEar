#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lists playback devices and plays synthesized speech to a chosen one (e.g. a virtual audio cable)."""
import sounddevice as sd


def list_output_devices():
    devices = sd.query_devices()
    return [
        {'index': index, 'name': device['name']}
        for index, device in enumerate(devices)
        if device['max_output_channels'] > 0
    ]


def play(pcm, sample_rate, device_index=None):
    if pcm is None or len(pcm) == 0:
        return
    sd.play(pcm, samplerate=sample_rate, device=device_index, blocking=True)
