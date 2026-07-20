#!/bin/bash
git config filter.strip-notebook-output.clean "bash \$(git rev-parse --show-toplevel)/strip-notebook.sh"
  
git config filter.strip-notebook-output.smudge cat

echo "Git filter for stripping notebook output installed."