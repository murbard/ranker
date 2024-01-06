#!/bin/bash

# Stop on first error
set -e

# Compile LaTeX document
pdflatex ranker.tex
biber ranker
pdflatex ranker.tex
pdflatex ranker.tex

echo "Compilation successful."

