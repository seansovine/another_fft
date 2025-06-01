# Developer Notes

## Assumptions checking and error handling

We aren't really doing either of these in this code, but both are necessary
to do eventually. For example, we assume RGBA, unsigned byte pixel format
in loaded images, and assume that our loading machinery handles that as
needed. We might consider other formats later.

## Add arguments & remove hard-coded paths

Output paths are hard-coded in several places.

## Image aspect ratio, resizing and display

In converting to power-of-two sizes we lose the original image aspect ratio.
We could resize back to the original size as a final step.

The `wgpu_grapher image` command currently displays all images with a square
aspect ratio; we should fix this.

## Layout of `image` image pixel matrices

We can see from the code below (from `buffer.rs`) that `image` stores image
pixel data row-major:


```rust
#[inline(always)]
fn pixel_indices_unchecked(&self, x: u32, y: u32) -> Range<usize> {
	let no_channels = <P as Pixel>::CHANNEL_COUNT as usize;
	// If in bounds, this can't overflow as we have tested that at construction!
	let min_index = (y as usize * self.width as usize + x as usize) * no_channels;
	min_index..min_index + no_channels
}
```
