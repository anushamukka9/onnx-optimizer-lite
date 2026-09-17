"""Allow ``python -m onnx_optimizer_lite`` as an alias for the ``onnx-opt`` CLI."""

from .cli import main

if __name__ == "__main__":
    main()
