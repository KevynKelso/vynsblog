#!/bin/bash

mkdir blog/$1
cp ./blog/hello-world/index.md ./blog/$1
nvim ./blog/$1/index.md
