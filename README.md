# Another FFT

This is a Rust crate implementing the Fast Fourier Transform.
It's called _another_ FFT because there are already crates with great
FFT implementations. This one is more of a personal project for experimenting with the
various FFT implementations that are out there and some applications. Who knows, maybe
someday it will find some niche that isn't filled by existing libraries. Or maybe not,
and that's fine too.

It currently has the FFT implementation from

```text
Press, et al., _Numerical Reciples_, Third Edition.
Cambridge University Press, 2007.
```

which the authors credit to N.M. Brenner.

## Image processing

I've implemented a 2D FFT and added an `image_processing` crate that provides several
operations, including the FFT, on image files.

Here are some examples of images with their FTs:

<p align="center" margin="20px">
        <img src="screenshots/example_1_fft.png" alt="drawing" width="800" style="padding-top: 15px; padding-bottom: 10px"/>
</p>

<p align="center" margin="20px">
        <img src="screenshots/example_2_fft.png" alt="drawing" width="800" style="padding-top: 15px; padding-bottom: 10px"/>
</p>

In fact the images shown are the magnitudes of the Fourier coefficients, which are in general complex,
and the images are converted to grayscale before the FT is taken.

__Filtering:__

I've also implemented a basic Fourier-space filter, as an example of the kinds of things that can be done.
This allows us to apply a multiplier to the Fourier coefficients of the image.
The [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem) says that this is equivalent
to convolving the image with a corresponding kernel, but some kernels are much easier to understand in
Fourier space. Like a high/low pass filter, for example.

Here is an example of a high-pass filter applied to an image:

<p align="center" margin="20px">
        <img src="screenshots/example_1_fft_filtered.png" alt="drawing" width="800" style="padding-top: 15px; padding-bottom: 10px"/>
</p>

We cut out the Fourier coefficients within a distance of 100 from the origin in Fourier space.
You can see that this affects the near-constant regions of the image, but leaves the regions with
more detailed features less changed. This is expected because Fourier coefficients close to the
origin represent more slowly-oscillating frequency components in the image.

One common application in image processing would be to add a multiple of the high-pass image back to
the original image to produce a sharpening effect.
