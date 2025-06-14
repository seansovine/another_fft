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

We've implemented a 2D FFT and added an `image_processing` crate that provides several
operations, including the FFT, on image files.

Here are some examples of images with their FFTs:

<p align="center" margin="20px">
        <img src="screenshots/example_1_fft.png" alt="drawing" width="800" style="padding-top: 15px; padding-bottom: 10px"/>
</p>

<p align="center" margin="20px">
        <img src="screenshots/example_2_fft.png" alt="drawing" width="800" style="padding-top: 15px; padding-bottom: 10px"/>
</p>

In fact the images show are the magnitudes of the FFT coefficients, which are in general complex,
and the images are converted to grayscale before the FFT is taken.
