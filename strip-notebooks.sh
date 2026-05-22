#!/bin/bash
exec jupyter nbconvert \
  --ClearOutputPreprocessor.enabled=True \
  --ClearMetadataPreprocessor.enabled=True \
  --to=notebook --stdin --stdout --log-level=ERROR