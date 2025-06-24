#!/bin/bash

# Script to connect local repository to GitHub
# Usage: ./setup_github.sh <your-github-repo-url>

if [ $# -eq 0 ]; then
    echo "Usage: ./setup_github.sh <your-github-repo-url>"
    echo "Example: ./setup_github.sh https://github.com/yourusername/kpi-analytics.git"
    exit 1
fi

GITHUB_URL=$1

echo "Setting up GitHub connection..."
echo "Repository URL: $GITHUB_URL"

# Add the remote origin
git remote add origin $GITHUB_URL

# Push to GitHub
echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "Setup complete! Your repository is now connected to GitHub."
echo "Future changes can be pushed with: git push" 