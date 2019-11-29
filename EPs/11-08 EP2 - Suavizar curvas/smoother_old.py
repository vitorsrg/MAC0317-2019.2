#!/usr/bin/env python3.7


"""
Smooth digital curves.
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



import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib import colorbar
from matplotlib.colors import LogNorm, Normalize, to_rgb

from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image

from scipy.fftpack import dct, idct

mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

word = 2**8-1
norml8 = Normalize(0, word)

img = np.asarray(Image.open('./fruits/pineapples.1.jpg').convert('L'), dtype=float)

img = 255*np.array([
    [0, 0, 0, 0, 0, 0,],
    [0, 0, 1, 1, 1, 0,],
    [0, 1, 0, 0, 1, 0,],
    [0, 1, 1, 1, 1, 0,],
    [0, 1, 1, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0,],
])

img = np.asarray(Image.open('./fox_binary.png').convert('L'), dtype=float)


import cv2 as cv

def add_colorbar(ax, pos='right', size=0.1, pad=0.05, 
                 cmap=None, norm=None, off=False, 
                 orientation='vertical', sharex=None):
    divider = make_axes_locatable(ax)
    bar = divider.append_axes(pos, size, pad=pad, sharex=sharex)
    
    if isinstance(cmap, str):
        cmap = matplotlib.cm.cmap_d[cmap]

    if off:
        bar.axis('off')
    else:
        colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, 
                              orientation=orientation)

    return bar

def drawimg(ax, title, img, cmap, norm):
    ax.set_title(title)
    ax.imshow(img, cmap=cmap, norm=norm)
    add_colorbar(ax, cmap=cmap, norm=norm)

def gridplot(shape, factor=6):
    shape = np.array(shape)

    fig, ax = plt.subplots(*shape, figsize=(factor*shape)[::-1])
    ax = ax.reshape(shape)
    fig.set_tight_layout(True)

    return ax

ax = gridplot((3, 3), factor=3)
# ret, thresh = cv.threshold(img, 127, 255, 0)
# print(ret)
# print(thresh)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

drawimg(ax[0, 0], 'Original Image', img, 'gray', norml8)

import scipy.ndimage
# img = 255*(img<=np.mean(img))
# img = 255*scipy.ndimage.binary_closing(img)

drawimg(ax[0, 1], 'Original Image', img, 'gray', norml8)

from skimage import measure

contours = measure.find_contours(img, 0)
print(contours)

for n, contour in enumerate(contours[:1]):
    ax[0, 1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='purple')


# img = scipy.ndimage.filters.laplace(img, mode='constant', cval=0)
# img = 255*(img>np.mean(img))

# img[:, :] = 0

# for i, j in contours[0].astype(int):
#     img[i, j] = 255

drawimg(ax[0, 2], 'Original Image', img, 'gray', norml8)

def first(img):
    i0, j0 = 0, 0

    H, W = img.shape
    for j in range(W):
        for i in range(H):
            if img[i, j] == 255:
                return i, j

    return None

def contour(img):
    i, j = first(img)
    h, w = img.shape

    visited = np.full(img.shape, False, dtype=bool)
    directions = np.array([
        (-1, -1), (-1,  0), (-1,  1),   # 0 1 2
        ( 0, -1), ( 0,  0), ( 0,  1),   # 3 4 5
        ( 1, -1), ( 1,  0), ( 1,  1),   # 6 7 8
    ])
    order = [7, 8, 5, 2, 1, 0, 3, 6]
    polygon = []

    k0 = 0
    while True:
        if visited[i, j]:
            break
        
        visited[i, j] = True
        polygon.append((i, j))

        for k in range(8):
            ni, nj = directions[order[(k0+k)%8]] + (i, j)

            if    ni < 0 or ni >= h \
               or nj < 0 or nj >= w:
                continue
            if img[ni, nj] != 255:
                continue

            i, j = ni, nj
            k0 = (k0+k+5)%8
            break

    return np.array(polygon)

poly = contour(img)

# for l in range(poly.shape[0]):
#     img[tuple(poly[l])] = 100


drawimg(ax[1, 0], 'Original Image', img, 'gray', norml8)
ax[1, 0].plot(poly[:, 1], poly[:, 0], color='green')

poly = contours[0]

h, w = img.shape

ax[1, 1].plot(np.mgrid[0:1:poly.shape[0]*1j], poly[:, 0].flatten()/h, color='red')
ax[1, 1].plot(np.mgrid[0:1:poly.shape[0]*1j], poly[:, 1].flatten()/w, color='blue')

fpoly = np.zeros(poly.shape, dtype=complex)

fpoly[:, 0] = np.fft.fft(poly[:, 0].flatten())
fpoly[:, 1] = np.fft.fft(poly[:, 1].flatten())

ax[1, 2].plot(np.mgrid[0:1:poly.shape[0]*1j], np.log1p(np.abs(fpoly[:, 0].flatten()/h)), color='red')
ax[1, 2].plot(np.mgrid[0:1:poly.shape[0]*1j], np.log1p(np.abs(fpoly[:, 1].flatten()/w)), color='blue')

p = 0.99

fpoly[:, 0] = fpoly[:, 0]*((np.mgrid[0:1:poly.shape[0]*1j]<(1-p)/2)|(np.mgrid[0:1:poly.shape[0]*1j]>(1+p)/2))
fpoly[:, 1] = fpoly[:, 1]*((np.mgrid[0:1:poly.shape[0]*1j]<(1-p)/2)|(np.mgrid[0:1:poly.shape[0]*1j]>(1+p)/2))

ax[2, 0].plot(np.mgrid[0:1:poly.shape[0]*1j], np.log1p(np.abs(fpoly[:, 0].flatten()/h)), color='red')
ax[2, 0].plot(np.mgrid[0:1:poly.shape[0]*1j], np.log1p(np.abs(fpoly[:, 1].flatten()/w)), color='blue')


poly[:, 0] = np.fft.ifft(fpoly[:, 0].flatten())
poly[:, 1] = np.fft.ifft(fpoly[:, 1].flatten())


ax[2, 1].plot(np.mgrid[0:1:poly.shape[0]*1j], poly[:, 0].flatten()/h, color='red')
ax[2, 1].plot(np.mgrid[0:1:poly.shape[0]*1j], poly[:, 1].flatten()/w, color='blue')


# for l in range(poly.shape[0]):
#     img[tuple(poly[l])] = 150

drawimg(ax[2, 2], 'Original Image', img, 'gray', norml8)
ax[2, 2].plot(poly[:, 1], poly[:, 0], color='purple')

plt.show()
exit()

if __name__ == '__main__':
    from argh import ArghParser

    parser = ArghParser()
    parser.add_commands(basecli)
    parser.dispatch()
