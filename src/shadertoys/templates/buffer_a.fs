// Bad Apple NN - Buffer A: Weight Storage
// This buffer stores the neural network weights as a texture
//
// ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4)
//
// Copyright © 2026 Jonathan Tremesayques
//
// This is anti-capitalist software, released for free use by individuals and
// organizations that do not operate by capitalist principles.
//
// Permission is hereby granted, free of charge, to any person or organization
// (the "User") obtaining a copy of this software and associated documentation
// files (the "Software"), to use, copy, modify, merge, distribute, and/or sell
// copies of the Software, subject to the following conditions:
//
//   1. The above copyright notice and this permission notice shall be included
//      in all copies or modified versions of the Software.
//
//   2. The User is one of the following:
//     a. An individual person, laboring for themselves
//     b. A non-profit organization
//     c. An educational institution
//     d. An organization that seeks shared profit for all of its members, and
//        allows non-members to set the cost of their labor
//
//   3. If the User is an organization with owners, then all owners are workers
//     and all workers are owners with equal equity and/or equal vote.
//
//   4. If the User is an organization, then the User is not law enforcement or
//      military, or working for or under either.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY
// KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
const int TOTAL_WEIGHTS = {{ total_weights }};
const int TEXTURE_SIZE = {{ tex_size }};

// Neural network weights (embedded directly in code)
const float NN_WEIGHTS[{{ total_weights }}] = float[{{ total_weights }}](
{%- for w in weights %}
    {{ w }}{% if not loop.last %},{% endif %}
{%- endfor %}
);

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Convert pixel coordinate to linear weight index
    int px = int(fragCoord.x);
    int py = int(fragCoord.y);
    int pixel_idx = py * TEXTURE_SIZE + px;
    int weight_idx = pixel_idx * 4;
    
    // Pack 4 weights per pixel (RGBA)
    vec4 packed = vec4(0.0);
    for (int i = 0; i < 4; i++) {
        int idx = weight_idx + i;
        if (idx < TOTAL_WEIGHTS) {
            // Normalize to [0, 1] for storage
            // Assuming weights are roughly in [-1, 1] range
            packed[i] = NN_WEIGHTS[idx] * 0.5 + 0.5;
        }
    }
    
    fragColor = packed;
}
