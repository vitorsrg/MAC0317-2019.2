#!/usr/bin/env python3.7


"""
Smooth digital curves.
"""


import numpy as np
import os
import sys

from PIL import Image


__author__      = 'Vitor Santa Rosa Gomes'
__copyright__   = 'Copyright 2019, Vitor Santa Rosa Gomes'
__credits__     = ['Vitor Santa Rosa Gomes']
__license__     = 'MIT'
__version__     = '1.0'
__maintainer__  = 'Vitor Santa Rosa Gomes'
__email__       = 'vitorssrg@gmail.com'
__status__      = 'Development'


def read_image(imgfile, channel):
    """read a channel from a image file"""
    
    if channel in {'RGB', 'RGBA'}:
        return np.array(Image.open(imgfile).convert(channel), dtype=int)

    if not isinstance(channel, int):
        channel = 'RGBAL'.index(channel)

    img = Image.open(imgfile).convert('RGBA')

    if channel < 4:
        return np.moveaxis(np.array(img, dtype=float), 
                           [0, 1, 2], 
                           [1, 2, 0])[channel]


    return np.array(img.convert('L'), dtype=float)

def contour_image(img, level):
    """countour a level curve from a channel"""
    h, w = img.shape

    def first(img):
        i0, j0 = 0, 0

        for j in range(w):
            for i in range(h):
                if img[i, j] == level:
                    return i, j

        return None, None

    i, j = first(img)
    if i is None and j is None:
        return []

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
            if img[ni, nj] != level:
                continue

            i, j = ni, nj
            k0 = (k0+k+5)%8
            break

    return np.array(polygon, dtype=int)

def save_image(img, path):
    """countour a level curve from a channel"""
    
    Image.fromarray(img).convert('RGB').save(path)

def main(imgrelpath, channel, intensity, threshold, ctrrelpath, ctrfltrelpath):
    """smooth digital curves"""

    img   = read_image(imgrelpath, channel)
    poly  = contour_image(img, intensity)

    cimg  = 0*img
    fcimg = 0*img

    if len(poly):
        fpoly = np.fft.fftshift(np.fft.fft(poly.astype(float), axis=0), axes=0)
        H, W  = img.shape
        L     = poly.shape[0]
        p     = np.abs(threshold/100)

        cimg[poly[:, 0].flatten(), poly[:, 1].flatten()] = 255

        mask  = np.mgrid[0:1:L*1j, 0:2][0]
        if threshold > 0:
            ffpoly = fpoly*((mask>=1/2-p/2)&(mask<=1/2+p/2))
        else:
            ffpoly = fpoly*((mask<=p/2)    |(mask>=1-p/2)  )

        fipoly = np.real(np.fft.ifft(np.fft.ifftshift(ffpoly, axes=0), 
                                    axis=0)
                        ).astype(int) % (H, W)

        fcimg[fipoly[:, 0].flatten(), fipoly[:, 1].flatten()] = 255

    save_image(cimg, ctrrelpath)
    save_image(fcimg, ctrfltrelpath)


if __name__ == '__main__':
    import sys
    import inspect

    try:
        imgrelpath      = sys.argv[1]

        channel         = int(sys.argv[2])
        intensity       = int(sys.argv[3])
        threshold       = float(sys.argv[4])

        ctrrelpath      = sys.argv[5]
        ctrfltrelpath   = sys.argv[6]
    except:
        helpstr = """
        usage: main.py  imgrelpath channel intensity threshold 
                        ctrrelpath ctrfltrelpath

        {}

        positional arguments:
        imgrelpath      image file
        channel         RGBA channel to work with (in 0..3)
        intensity       pixel intensity to contour (in 0..255)
        threshold       frequency threshold to filter 
                        (in -100..-1 or 1..100)
        ctrrelpath      contour detected from the original image
        ctrfltrelpath   filtered contour
        """
        helpstr = inspect.cleandoc(helpstr)
        helpstr = helpstr.format(inspect.cleandoc(__doc__))
        helpstr = inspect.cleandoc(helpstr)

        print(helpstr)
        exit()

    main(imgrelpath, channel, intensity, threshold, ctrrelpath, ctrfltrelpath)
