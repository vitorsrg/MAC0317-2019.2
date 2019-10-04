#!/usr/bin/env python3.7


"""
Digital sound synthesizer with linear ADSR profile.
"""

import os
import sys


__author__      = 'Vitor Santa Rosa Gomes'
__copyright__   = 'Copyright 2019, Vitor Santa Rosa Gomes'
__credits__     = ['Vitor Santa Rosa Gomes']
__license__     = 'MIT'
__version__     = '1.0'
__maintainer__  = 'Vitor Santa Rosa Gomes'
__email__       = 'vitorssrg@gmail.com'
__status__      = 'Development'

#region global
basecli = []
#endregion

#region sound utils
def note2freq(note):
    """
    convert a musical note to its frequency in hertz

    https://en.wikipedia.org/wiki/Scientific_pitch_notation
    """
    import re

    notes = ['C', 'C#', 
             'D', 'D#', 
             'E', 
             'F', 'F#', 
             'G', 'G#', 
             'A', 'A#', 
             'B']

    groups = re.search(r'^(?P<tone>[ABCDEFG])'
                       r'(?P<step>[\#b])?'
                       r'(?P<octave>-?\d)?$', note).groupdict()

    tone, step, octave = groups['tone'], \
                         groups['step'], \
                         groups['octave']

    offset = notes.index(tone) + 12*(int(octave or 4) - 4)
    if groups['step'] == 'b':
        offset -= 1
    elif groups['step'] == '#':
        offset += 1

    return 440*2.0**((offset-9)/12)

def midi2freq(note):
    """
    convert a midi note to its frequency in hertz

    https://en.wikipedia.org/wiki/Scientific_pitch_notation
    """

    return 440*2.0**((note-69)/12)

def soundwave(freq, seconds, samplerate):
    """
    generate a sound wave
    """
    import numpy as np

    support = np.linspace(0, seconds, samplerate*seconds, 
                          dtype=float, endpoint=False)

    return np.sin(2*np.pi*freq*support)

def playwav(filename, volume=0.10):
    """
    calls ffplay

    https://ffmpeg.org/
    """
    import subprocess
    import sys

    bold    = '\033[1m' if sys.stdout.isatty() else ''
    default = '\033[0m' if sys.stdout.isatty() else ''

    cmd = f"ffplay -autoexit -af volume={volume} '{filename}' > /dev/null 2>&1"
    
    print(f"{bold}{cmd}{default}")
    subprocess.run(['bash', '-c', cmd])

basecli.append(note2freq)
basecli.append(midi2freq)
basecli.append(playwav)
#endregion

#region file utils
def readpart(partfile):
    """
    convert a part file into a sheet music
    """
    import numpy as np
    import re

    sheet = []

    with open(partfile, 'r') as partreader:
        offset = 0

        for line in partreader.readlines()[1:]:
            line = re.split(r'\s+', line.strip())
            notes, duration = map(note2freq, line[:-1]), float(line[-1])/1000

            for note in notes:
                sheet.append((note, offset, duration, 1))

                offset += duration

    return np.array(sheet)

def readmidi(midifile):
    """
    convert a midi file into a sheet music
    """
    import numpy as np

    from mido import MidiFile


    sheet   = []
    offset  = 0
    started = dict()
    tempo   = 120

    midireader = MidiFile(midifile, clip=True)

    for msg in midireader:
        if 'time' in vars(msg):
            offset += msg.time
        if   msg.type == 'note_on' and msg.velocity != 0:
            started.setdefault(msg.channel, dict())[msg.note] = offset
        elif (msg.type == 'note_off' \
              or (msg.type == 'note_on' \
                  and msg.velocity == 0)) \
             and msg.channel in started \
             and msg.note in started[msg.channel]:
            sheet.append((midi2freq(msg.note), 
                          started[msg.channel][msg.note],
                          offset-started[msg.channel][msg.note],
                          1))
            started[msg.channel].pop(msg.note)
        elif msg.type == 'set_tempo':
            tempo = msg.tempo

    sheet = np.array(sheet)
    sheet[:, 1:3] = sheet[:, 1:3]*(tempo/midireader.ticks_per_beat)/10**3

    return sheet

def readadsr(adsrfile):
    """
    read an adsr into a transformer
    """
    import numpy as np

    from scipy import interpolate

    data = np.loadtxt(adsrfile)
    bars, volume = [0]+list(data[:, 0]), \
                   [0]+list(data[:, 1])

    assert len(bars) == 5
    assert len(volume) == 5

    bars = np.cumsum(bars)/np.sum(bars)
    volume = volume/np.max(volume)
    linear = interpolate.interp1d(bars, volume)

    def transformer(arr):
        wrapper = np.linspace(0, 1, len(arr), endpoint=False)
        return linear(wrapper)

    return transformer

def sheet2wav(sheet, wrapper, wavfile, samplerate=44100):
    """
    write a sheet to a wav file
    """
    import numpy as np
    import scipy.io.wavfile as Wavfile


    length     = np.max(sheet[:, 1] + sheet[:, 2])
    channel    = 0*soundwave(0, length, samplerate)
    concurrent = 0*soundwave(0, length, samplerate)
    end        = int(length*samplerate)

    for freq, offset, duration, volume in sheet:
        wave = soundwave(freq, duration, samplerate)
        wave = wrapper(wave)*wave

        start = int(offset*samplerate)
        delta = min(int(duration*samplerate), end-start)
        channel[start:start+delta] += volume*wave[:delta]

        concurrent[start:start+delta] += 1

    channel = np.divide(channel, concurrent, 
                        out=0*channel, 
                        where=concurrent!=0)
    channel = np.array((channel+1)/2*2**16 - 2**15)
    channel = np.min([channel, (2**15-1)*(0*channel+1)], axis=0)
    channel = np.max([channel, -2**15*(0*channel+1)], axis=0)
    channel = np.array([channel, channel], dtype=np.int16).T
    # channel = np.array([channel, channel], dtype=float).T

    Wavfile.write(wavfile, samplerate, channel)

def part2wav(adsrfile, partfile, wavfile, samplerate=44100):
    """
    create a wav file from adsr and part
    """

    wrapper = readadsr(adsrfile)
    sheet   = readpart(partfile)

    sheet2wav(sheet, wrapper, wavfile, samplerate)

def midi2wav(adsrfile, midifile, wavfile, samplerate=44100):
    """
    create a wav file from adsr and midi
    """

    wrapper = readadsr(adsrfile)
    sheet   = readmidi(midifile)

    sheet2wav(sheet, wrapper, wavfile, samplerate)

basecli.append(readpart)
basecli.append(readmidi)
basecli.append(part2wav)
basecli.append(midi2wav)
#endregion


if __name__ == '__main__':
    from argh import ArghParser

    parser = ArghParser()
    parser.add_commands(basecli)
    parser.dispatch()
