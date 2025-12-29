# GitHub Repository Setup

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `tennis-analysis-integration`
3. Description: `Unified system integrating shot detection with 3D player reconstruction for tennis match analysis`
4. Visibility: Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Push to GitHub

Run these commands in order:

```bash
# Make sure you're in the project directory
cd /home/saeed/MyProjects/sportai/tennis

# Add all files
git add .

# Commit (if not already committed)
git commit -m "Initial commit: Tennis analysis integration system

- Integration script for shot detection + 3D reconstruction
- Unified web viewer with FastAPI backend and React frontend
- Complete documentation and setup guides
- WASD controls for 3D scene navigation
- H.264 video encoding for browser compatibility"

# Rename branch to main (if needed)
git branch -M main

# Add remote (replace with your actual username if different)
git remote add origin https://github.com/sssabet/tennis-analysis-integration.git

# Push to GitHub
git push -u origin main
```

## Alternative: Using SSH

If you prefer SSH:

```bash
git remote add origin git@github.com:sssabet/tennis-analysis-integration.git
git push -u origin main
```

## Step 3: Update README with Repository Links

After pushing, update the README.md to include links to the dependency repositories.

## Troubleshooting

**If you get authentication errors:**
- Use GitHub CLI: `gh auth login`
- Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

**If remote already exists:**
```bash
git remote remove origin
git remote add origin https://github.com/sssabet/tennis-analysis-integration.git
```

**If you need to force push (be careful!):**
```bash
git push -u origin main --force
```

