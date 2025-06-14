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
