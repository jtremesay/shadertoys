// Bad Apple NN - Image: Neural Network Inference
// Reads weights from Buffer A (iChannel0) and performs forward pass

const int VIDEO_WIDTH = {{ width }};
const int VIDEO_HEIGHT = {{ height }};
const int TOTAL_FRAMES = {{ total_frames }};
const int TEXTURE_SIZE = {{ tex_size }};

// Read weight from Buffer A texture
float readWeight(int index) {
    int pixel_idx = index / 4;
    int channel = index % 4;
    
    int tex_y = pixel_idx / TEXTURE_SIZE;
    int tex_x = pixel_idx % TEXTURE_SIZE;
    
    vec2 uv = (vec2(tex_x, tex_y) + 0.5) / float(TEXTURE_SIZE);
    vec4 pixel = texelFetch(iChannel0, ivec2(tex_x, tex_y), 0);
    
    // Denormalize from [0, 1] to [-1, 1]
    float value = pixel[channel] * 2.0 - 1.0;
    return value;
}

// Activation functions
float relu(float x) {
    return max(0.0, x);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Neural network forward pass
float neuralNetwork(vec3 input_) {
    // Architecture: {{ hidden_sizes }} -> 1
    
    int offset = 0;
    {% for layer in layer_info[:-1] %}
    // Layer {{ loop.index0 }}: [{{layer["input"]}}] -> [{{layer["output"]}}]
    float hidden{{ loop.index }}[{{ layer["output"] }}];
    for (int i = 0; i < {{ layer["output"] }}; i++) {
        float sum = 0.0;
        for (int j = 0; j < {{ layer["input"] }}; j++) {
            sum += {% if loop.index0 == 0 %}input_[j]{% else %}hidden{{ loop.index0 }}[j]{% endif %} * readWeight(offset + i * {{ layer["input"] }} + j);
        }
        sum += readWeight(offset + {{ layer["input"] * layer["output"] }} + i);
        hidden{{ loop.index }}[i] = relu(sum);
    }
    offset += {{ layer["input"] * layer["output"] + layer["output"] }};
    {% endfor %}
    // Output layer: [{{ layer_info[-1]["input"] }}] -> [{{ layer_info[-1]["output"] }}]
    float output_ = 0.0;
    for (int i = 0; i < {{ layer_info[-1]["input"] }}; i++) {
        output_ += hidden{{ layer_info|length - 1 }}[i] * readWeight(offset + i);
    }
    output_ += readWeight(offset + {{ layer_info[-1]["input"] }});
    output_ = sigmoid(output_);
    
    return output_;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Calculate current frame based on time (30 fps)
    int frame = int(iTime * 30.0) % TOTAL_FRAMES;
    
    // Get pixel coordinates
    vec2 uv = fragCoord / iResolution.xy;
    int px = int(uv.x * float(VIDEO_WIDTH));
    int py = int((1.0 - uv.y) * float(VIDEO_HEIGHT));
    
    // Boundary check
    if (px >= VIDEO_WIDTH || py >= VIDEO_HEIGHT) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Normalize inputs to [0, 1]
    float frame_norm = float(frame) / float(TOTAL_FRAMES);
    float x_norm = float(px) / float(VIDEO_WIDTH);
    float y_norm = float(py) / float(VIDEO_HEIGHT);
    
    vec3 input_ = vec3(frame_norm, x_norm, y_norm);
    
    // Run neural network
    float gray = neuralNetwork(input_);
    
    fragColor = vec4(vec3(gray), 1.0);
}
