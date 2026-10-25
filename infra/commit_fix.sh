#!/usr/bin/env bash
# Stops tracking generated files, commits, then reports state.
set -e
cd ~/agritune
printf 'unsloth_compiled_cache/\nbench_results.jsonl\n' >> .gitignore
git rm -r -q --cached services/train/unsloth_compiled_cache services/train/bench_results.jsonl
git add -A
git commit -q -m "Ignore Unsloth compiled cache and local benchmark results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline
git status --short | wc -l
