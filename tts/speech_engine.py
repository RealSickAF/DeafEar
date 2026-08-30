#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Romanian text-to-speech via edge-tts neural voices, decoded to PCM with on-disk caching."""
import asyncio
import hashlib
import io
import os

import av
import edge_tts
import numpy as np

VOICE_MALE = 'ro-RO-EmilNeural'
VOICE_FEMALE = 'ro-RO-AlinaNeural'

CACHE_DIR = os.path.join('tts', 'cache')


def _cache_path(text, voice):
    key = hashlib.sha1(f'{voice}::{text}'.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f'{key}.npy')


async def _synthesize_mp3(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            chunks.extend(chunk['data'])
    return bytes(chunks)


def _decode_mp3(mp3_bytes, target_rate=24000):
    container = av.open(io.BytesIO(mp3_bytes))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format='flt', layout='mono', rate=target_rate)

    chunks = []
    for frame in container.decode(stream):
        resampled = resampler.resample(frame)
        frames = resampled if isinstance(resampled, list) else [resampled]
        for out_frame in frames:
            if out_frame is not None:
                chunks.append(out_frame.to_ndarray())

    if not chunks:
        return np.zeros(0, dtype=np.float32), target_rate

    pcm = np.concatenate(chunks, axis=1).flatten().astype(np.float32)
    return pcm, target_rate


def synthesize(text, gender='female'):
    """Returns (pcm_float32_mono, sample_rate) for the given Romanian text, using a disk cache."""
    voice = VOICE_FEMALE if gender == 'female' else VOICE_MALE
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_file = _cache_path(text, voice)
    if os.path.exists(cache_file):
        cached = np.load(cache_file, allow_pickle=True).item()
        return cached['pcm'], cached['rate']

    mp3_bytes = asyncio.run(_synthesize_mp3(text, voice))
    pcm, rate = _decode_mp3(mp3_bytes)

    np.save(cache_file, {'pcm': pcm, 'rate': rate})
    return pcm, rate
