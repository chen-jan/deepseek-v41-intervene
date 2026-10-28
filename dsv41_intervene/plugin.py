"""``vllm.general_plugins`` entry point.

Does nothing unless ``DSV41_ENGINE=1``, so the plain server never gets patched even if this package is
installed in its venv (it must not be; see docs/INTERFACE.md section 0, item 2).
"""

import os


def register() -> None:
    """Apply ``patch.apply()`` in this process when ``DSV41_ENGINE=1``. Safe to call more than once.

    ``patch.apply`` touches only the DeepSeek V4.1 classes, so other models are unaffected. The installed layers
    (``DSV41_EDIT_LAYERS``) and capture layers are validated here, so a bad value fails the boot before the model
    loads instead of after.
    """
    if os.environ.get("DSV41_ENGINE") != "1":
        return
    from . import capture_layers, edit_layers, patch

    capture_layers(edit_layers())
    patch.apply()
