#!/bin/bash
# DeepCollector DGX to GitHub Sync Script

# Define your paths (Adjust WORKSPACE_DIR if your main files live elsewhere)
WORKSPACE_DIR="$HOME/Desktop/DeepKG"
REPO_DIR="$WORKSPACE_DIR/deepcollector"
TARGET_DIR="$REPO_DIR/localdgxfiles"

echo "========================================================"
echo "🔄 INITIALIZING GITHUB SYNC FROM DGX"
echo "========================================================"

# 1. Ensure the Git repository and target subdirectory exist
if [ ! -d "$REPO_DIR" ]; then
    echo "❌ Error: Git repository directory $REPO_DIR not found."
    exit 1
fi

mkdir -p "$TARGET_DIR"

# 2. Copy the active scripts into the Git tracking folder
echo "📥 Copying .py and .sh files to $TARGET_DIR..."

# This finds only the .py and .sh files in your main workspace (avoiding deep subfolders) 
# and copies them directly into the localdgxfiles subdirectory.
find "$WORKSPACE_DIR" -maxdepth 1 -type f \( -name "*.py" -o -name "*.sh" \) -exec cp {} "$TARGET_DIR/" \;

echo "✅ Local file copy complete."

# 3. Navigate into the Git Repository
cd "$REPO_DIR" || exit 1

# 4. Stage, Commit, and Push
echo -e "\n🖥️ Staging changes for Git..."
git add "localdgxfiles/"

# Check if there are actual changes to commit to avoid empty commit errors
if git diff-index --quiet HEAD --; then
    echo "ℹ️ No changes detected. GitHub is already up to date."
else
    # Create a dynamic timestamp for the commit message
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M")
    COMMIT_MSG="Auto-update local DGX scripts - $TIMESTAMP"
    
    echo "📝 Committing changes: '$COMMIT_MSG'"
    git commit -m "$COMMIT_MSG"
    
    echo "🚀 Pushing updates to GitHub..."
    # Note: Change 'main' to 'master' below if your branch uses the older naming convention
    git push origin main 
    
    echo -e "\n========================================================"
    echo "✅ GITHUB UPDATE SUCCESSFUL"
    echo "========================================================"
fi
