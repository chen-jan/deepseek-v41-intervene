#!/bin/bash
# Create the server's API key file ($DSV41_KEY_FILE, default $DSV41_ROOT/.serve_key) with mode 0600, if missing.
# The key is never printed. Servers read it into VLLM_API_KEY; tools read it from the file.
set -euo pipefail
. "$(dirname "$0")/common.sh" || exit 2
if [ -s "$DSV41_KEY_FILE" ]; then echo "key file exists: $DSV41_KEY_FILE"; exit 0; fi
umask 077
mkdir -p "$(dirname "$DSV41_KEY_FILE")"
python3 -c "import secrets; print(secrets.token_hex(32))" > "$DSV41_KEY_FILE"
chmod 600 "$DSV41_KEY_FILE"
echo "wrote $DSV41_KEY_FILE (0600)"
