#!/bin/bash
# Script to build and install the modified sgl-router with X-SGLang-Worker header support

set -e

echo "Building and installing modified sgl-router..."

# Check if we're in the right directory
if [ ! -d "sgl-router" ]; then
    echo "Error: sgl-router directory not found!"
    echo "Please run this script from the sglang root directory"
    exit 1
fi

cd sgl-router

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo not found!"
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "Building sgl-router with cargo..."
cargo build --release

echo "Installing sgl-router..."
pip install -e .

echo "Verifying installation..."
python -c "import sglang_router; print('sglang_router version:', sglang_router.__version__ if hasattr(sglang_router, '__version__') else 'unknown')"

echo "Done! The modified router with X-SGLang-Worker header support is now installed."
echo ""
echo "To test GPU tracking, run:"
echo "  python sglang_test_framework/test_route.py"
echo ""
echo "The router will now include X-SGLang-Worker headers in responses,"
echo "enabling accurate GPU tracking in the routing tests."