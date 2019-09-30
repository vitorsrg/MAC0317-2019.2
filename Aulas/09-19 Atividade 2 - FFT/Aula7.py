from numpy import linspace, sin, cos, pi, absolute, real, imag, arctan, arctan2
from numpy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

def dft(x):
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)

## 
## Exercício 1
##
def exercicio1():

	# Sampling rate
	fs = 32  # Hz

	# Time is from 0 to 1 seconds, but leave off the endpoint, so
	# that 1.0 seconds is the first sample of the *next* chunk
	length = 1  # second
	N = fs * length
	t = linspace(0, length, num=N, endpoint=False)

	# Generate a sinusoid at frequency f
	f1 = 8  # Hz
	a = 3.0 * sin(2 * pi * f1 * t)

	# Plot signal, showing how endpoints wrap from one chunk to the next
	plt.subplot(3, 1, 1)
	plt.plot(t, a, '.-')
	plt.plot(t, a, 'r.')  # first sample of next chunk
	plt.margins(0.1, 0.1)
	plt.xlabel('Time [s]')

	## DFT
	ffta = fft(a)

	## normalized amplitude
	ampl = 2 * 1/N * abs(ffta)

	## real and imaginary part
	re = real(ffta)
	im = imag(ffta)

	# FFT frequency bins
	freqs = fftfreq(N, 1/fs)

	# Plot shifted data on a shifted axis
	plt.subplot(3, 1, 2)
	plt.stem(fftshift(freqs), fftshift(ampl))
	plt.margins(0.1, 0.1)
	plt.xlabel('Frequency [Hz]')

	plt.subplot(3, 1, 3)
	plt.stem(fftshift(freqs), fftshift(arctan2(im, re)*180/pi))
	plt.margins(0.1, 0.1)
	plt.xlabel('Frequency [Hz]')
	plt.tight_layout()
	plt.show()

## 
## Exercício 2
##
def exercicio2():
	
	# Sampling rate
	fs = 64  # Hz

	# Time is from 0 to 1 seconds, but leave off the endpoint, so
	# that 1.0 seconds is the first sample of the *next* chunk
	length = 1  # second
	N = fs * length
	t = linspace(0, length, num=N, endpoint=False)

	# Generate a sinusoids
	f1 = 8   # Hz
	f2 = 12  # Hz
	f3 = 20  # Hz
	a = 3.0 * sin(2 * pi * f1 * t) + 1.5 * cos(2 * pi * f2 * t) + 0.5 * sin(2 * pi * f3 * t) 

	# Plot signal, showing how endpoints wrap from one chunk to the next
	plt.subplot(3, 1, 1)
	plt.plot(t, a, '.-')
	plt.plot(t, a, 'r.')  # first sample of next chunk
	plt.margins(0.1, 0.1)
	plt.xlabel('Time [s]')

	## DFT
	ffta = fft(a)

	## normalized amplitude
	ampl = 2 * 1/N * abs(ffta)

	## real and imaginary part
	re = real(ffta)
	im = imag(ffta)

	# FFT frequency bins
	freqs = fftfreq(N, 1/fs)

	# Plot shifted data on a shifted axis
	plt.subplot(3, 1, 2)
	plt.stem(fftshift(freqs), fftshift(ampl))
	plt.margins(0.1, 0.1)
	plt.xlabel('Frequency [Hz]')

	plt.subplot(3, 1, 3)
	plt.stem(fftshift(freqs), fftshift(arctan2(im, re)*180/pi))
	plt.margins(0.1, 0.1)
	plt.xlabel('Frequency [Hz]')
	plt.tight_layout()
	plt.show()


