#!/bin/bash

echo "Setting up Git repository connection and pushing changes..."

# Check if we're already in a git repository
if [ ! -d .git ]; then
  echo "Git repository not initialized. Initializing now..."
  git init
  
  # Check if remote origin exists
  if ! git remote | grep -q "origin"; then
    echo "Adding remote origin..."
    git remote add origin https://github.com/UliKonstantin/SubspaceNet_Update.git
  fi
else
  echo "Git repository already initialized."
  
  # Check if remote origin exists and update it if needed
  if ! git remote | grep -q "origin"; then
    echo "Adding remote origin..."
    git remote add origin https://github.com/UliKonstantin/SubspaceNet_Update.git
  else
    echo "Ensuring remote origin has the correct URL..."
    git remote set-url origin https://github.com/UliKonstantin/SubspaceNet_Update.git
  fi
fi

# Stage all changes
echo "Staging all changes..."
git add .

# Prompt for commit message
echo "Please enter a commit message:"
read commit_message

# Commit changes
echo "Committing changes..."
git commit -m "$commit_message"

# Push changes
echo "Pushing changes to remote repository..."
git push -u origin main

echo "Done! Your changes have been pushed to https://github.com/UliKonstantin/SubspaceNet_Update" 