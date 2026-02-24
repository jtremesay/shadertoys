# ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4)
#
# Copyright © 2026 Jonathan Tremesayques
#
# This is anti-capitalist software, released for free use by individuals and
# organizations that do not operate by capitalist principles.
#
# Permission is hereby granted, free of charge, to any person or organization
# (the "User") obtaining a copy of this software and associated documentation
# files (the "Software"), to use, copy, modify, merge, distribute, and/or sell
# copies of the Software, subject to the following conditions:
#
#   1. The above copyright notice and this permission notice shall be included
#      in all copies or modified versions of the Software.
#
#   2. The User is one of the following:
#     a. An individual person, laboring for themselves
#     b. A non-profit organization
#     c. An educational institution
#     d. An organization that seeks shared profit for all of its members, and
#        allows non-members to set the cost of their labor
#
#   3. If the User is an organization with owners, then all owners are workers
#     and all workers are owners with equal equity and/or equal vote.
#
#   4. If the User is an organization, then the User is not law enforcement or
#      military, or working for or under either.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY
# KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

YT_URL = 'https://www.youtube.com/watch?v=FtutLA63Cp8'
YT_FORMAT = 243
FFMPEG_SCALE = 60:40
FFMPEG_START = 0:00:00
FFMPEG_DURATION = 14
NN_EPOCHS = 10_000

all: gen-shader
.PHONY: all

dl-video: original.webm
.PHONY: dl-video

original.webm:
	yt-dlp \
		--remote-components ejs:github \
		-f $(YT_FORMAT) \
		$(YT_URL) \
		-o $@

gen-sample: video.webm
.PHONY: gen-sample

video.webm: original.webm
	ffmpeg -i $< -ss $(FFMPEG_START) -t $(FFMPEG_DURATION) -vf scale=$(FFMPEG_SCALE) -y $@

extract-pixels: video_pixels.parquet
.PHONY: extract-pixels

video_pixels.parquet: video.webm
	shadertoys_extract_pixels


train-nn: nn_weights.npz
.PHONY: train-nn

nn_weights.npz: video_pixels.parquet
	shadertoys_train_nn --epochs $(NN_EPOCHS)

gen-shader: shadertoy_buffer_a.fs
.PHONY: gen-shader

shadertoy_buffer_a.fs: nn_weights.npz
	shadertoys_generate_shaders

