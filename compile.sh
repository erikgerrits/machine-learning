#!/usr/bin/env bash
clear

# Create documentation
cd src/lib
typedoc --out ../../docs --module commonjs --target ES2017 .
cd ../..

# Compile TypeScript to JavaScript
rm -rf dist
echo "Compiling TypeScript";
tsc
echo "Done Compiling TypeScript";
