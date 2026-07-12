#!/bin/bash

echo "========================================================"
echo "🔄 RECONCILING LOCAL AND GITHUB REPOSITORIES"
echo "========================================================"

# 1. Dynamically hunt for the .git root directory
if [ -d ".git" ]; then
    GIT_ROOT=$(pwd)
elif [ -d "../.git" ]; then
    GIT_ROOT=$(dirname $(pwd))
elif [ -d "../../.git" ]; then
    GIT_ROOT=$(dirname $(dirname $(pwd)))
elif [ -d "deepcollector/.git" ]; then
    GIT_ROOT="$(pwd)/deepcollector"
else
    echo "❌ ERROR: Cannot find the hidden .git directory!"
    exit 1
fi

cd "$GIT_ROOT" || exit 1

# 2. Stage changes safely
echo "🖥️ Staging code and structural moves..."
# Stage all tracked files that were modified or deleted
git add -u
# Explicitly stage our new untracked scripts (ignoring Admin/, .env, and PDFs)
git add resume_run3.sh rescue*.py reorganize_files.py fetch_pdfgems.py 2>/dev/null

# 3. Commit locally BEFORE pulling
echo "📝 Committing local changes..."
git commit -m "Flattened execution architecture and script updates - $(date +'%Y-%m-%d %H:%M')"

# 4. Pull remote changes on top of our clean working tree
echo "⬇️ Pulling remote changes from GitHub..."
git pull --rebase

# 5. Push upstream
echo "🚀 Pushing updates to GitHub..."
if git push; then
    echo "========================================================"
    echo "✅ GITHUB RECONCILIATION SUCCESSFUL"
    echo "========================================================"
else
    echo "========================================================"
    echo "❌ GITHUB PUSH FAILED. Check terminal output for errors."
    echo "========================================================"
fi
