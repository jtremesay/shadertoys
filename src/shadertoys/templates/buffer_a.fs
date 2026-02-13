// Bad Apple NN - Buffer A: Weight Storage
// This buffer stores the neural network weights as a texture

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
