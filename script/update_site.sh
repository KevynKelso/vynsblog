#!/bin/bash

git add .
git commit -m "bot: weekly update"
git push
npm run publish

