#!/usr/bin/env bash
cd src/lib
typedoc --out ../../../mldoc --module commonjs --target ES2017 .
cd ../../mldoc
git add --all
git commit -a -m "Documentation update"
git push origin master