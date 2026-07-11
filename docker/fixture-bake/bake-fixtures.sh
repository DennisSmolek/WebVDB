#!/bin/sh
# Entrypoint: with no args, bake the primitive fixtures into /out/primitives.
# With args, exec them — exposes nanovdb_convert / nanovdb_print / etc.
set -eu

if [ "$#" -eq 0 ]; then
  exec bake_primitives /out/primitives
fi
exec "$@"
