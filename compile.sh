#!/usr/bin/env bash
clear

# Compile TypeScript to JavaScript
rm -rf dist
echo "Compiling TypeScript";
tsc
echo "Done Compiling TypeScript";
