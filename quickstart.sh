#!/usr/bin/env bash
# Quick start script for Bad Apple Shadertoy project

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Bad Apple → Shadertoy Neural Network Compression Pipeline    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if weights already exist
if [ -f "bad_apple/nn_weights_tiny.npz" ]; then
    echo "✓ Weights file found: bad_apple/nn_weights_tiny.npz"
    read -p "Do you want to retrain? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "→ Skipping training, using existing weights"
        SKIP_TRAINING=true
    else
        SKIP_TRAINING=false
    fi
else
    echo "→ No weights found, training required"
    SKIP_TRAINING=false
fi

# Step 1: Train NN (if needed)
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 1: Training Neural Network"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Architecture: Tiny [3→32→64→32→1] = 4,353 parameters"
    echo "Training: 5% sample, 30 epochs, batch 8192"
    echo "Estimated time: 1-2 hours (CPU) or 10-15 min (GPU)"
    echo ""
    
    python3 bad_apple/train_nn.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Training failed!"
        exit 1
    fi
    
    echo "✓ Training complete!"
else
    echo "→ Using existing weights"
fi

# Step 2: Generate Shadertoy shader
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Generating Shadertoy Multi-Pass Shader"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 generate_shadertoy_multipass.py bad_apple/nn_weights_tiny.npz

if [ $? -ne 0 ]; then
    echo "❌ Shader generation failed!"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Pipeline Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Generated Files:"
echo "   • bad_apple/shadertoy_buffer_a.glsl  - Buffer A shader"
echo "   • bad_apple/shadertoy_image.glsl     - Image shader"
echo "   • bad_apple/SHADERTOY_SETUP.md       - Setup instructions"
echo ""
echo "🌐 Next Steps:"
echo "   1. Go to: https://www.shadertoy.com/new"
echo "   2. Follow instructions in: bad_apple/SHADERTOY_SETUP.md"
echo "   3. Watch Bad Apple play in your browser!"
echo ""
echo "📖 For detailed info, see: README.md"
echo ""
