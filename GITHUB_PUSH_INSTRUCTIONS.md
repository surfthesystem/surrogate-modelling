# GitHub Push Instructions

Your code is now committed locally and ready to be pushed to GitHub. Follow these steps:

---

## Step 1: Create GitHub Repository (if you haven't already)

### Option A: Via GitHub Website

1. Go to https://github.com
2. Click the **+** icon in the top right, select **New repository**
3. Fill in:
   - **Repository name**: `surrogate-modelling` (or your preferred name)
   - **Description**: "Reservoir simulator with IMPES two-phase flow for ML surrogate modeling"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **Create repository**
5. Copy the repository URL (will be like `https://github.com/YOUR_USERNAME/surrogate-modelling.git`)

### Option B: Via GitHub CLI (if installed)

```bash
gh repo create surrogate-modelling --public --source=. --remote=origin
```

---

## Step 2: Add Remote and Push

In your project directory, run:

```bash
# Add your GitHub repository as the remote
git remote add origin https://github.com/YOUR_USERNAME/surrogate-modelling.git

# Verify remote was added
git remote -v

# Push to GitHub (main branch)
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (NOT your password)

### Creating a Personal Access Token:

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **Generate new token (classic)**
3. Name it "Surrogate Modelling Repo"
4. Select scopes: `repo` (all permissions)
5. Click **Generate token**
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing

---

## Step 3: Verify Push

Check your GitHub repository page:
- All files should be visible
- README.md should display on the main page
- Verify the commit message appears

---

## Step 4: Clone on Google Cloud VM

Once pushed to GitHub, you can clone on your GCP VM:

```bash
# On your GCP VM
git clone https://github.com/YOUR_USERNAME/surrogate-modelling.git
cd surrogate-modelling

# Follow the setup steps in GOOGLE_CLOUD_DEPLOYMENT.md
```

---

## Alternative: Use SSH Keys (Recommended for frequent pushes)

### Generate SSH key (on your local machine):

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location
# Optionally set a passphrase
```

### Add SSH key to GitHub:

```bash
# Copy the public key
cat ~/.ssh/id_ed25519.pub

# Or on Windows
type %USERPROFILE%\.ssh\id_ed25519.pub
```

1. Go to GitHub → Settings → SSH and GPG keys
2. Click **New SSH key**
3. Paste the key content
4. Click **Add SSH key**

### Use SSH remote:

```bash
# Remove HTTPS remote
git remote remove origin

# Add SSH remote
git remote add origin git@github.com:YOUR_USERNAME/surrogate-modelling.git

# Push
git push -u origin main
```

Now you won't need to enter credentials for each push!

---

## Common Issues and Solutions

### Issue: "remote origin already exists"

```bash
# Remove existing remote
git remote remove origin

# Add your remote
git remote add origin https://github.com/YOUR_USERNAME/surrogate-modelling.git
```

### Issue: "rejected" or "non-fast-forward"

This means the remote has changes you don't have locally.

```bash
# Pull first (if this is a new repo, this shouldn't happen)
git pull origin main --allow-unrelated-histories

# Then push
git push -u origin main
```

### Issue: Large file warnings

If you get warnings about large files (>50 MB):

```bash
# Check file sizes
du -sh data/* results/*

# If needed, add large files to .gitignore
echo "data/*.npy" >> .gitignore
echo "results/*.npz" >> .gitignore

# Commit the updated .gitignore
git add .gitignore
git commit -m "Update .gitignore for large files"
git push
```

Consider using **Git LFS** for large data files:

```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.npy"
git lfs track "*.npz"
git lfs track "*.h5"

# Commit .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

---

## What's Included in Your Push

### Files Committed (359 files):
- ✓ All documentation (.md files)
- ✓ Simulator code (simulator/ - 19 Python files)
- ✓ Source code (src/ - Phase 1 generation)
- ✓ Data files (data/ - 476 KB)
- ✓ Results (results/ - 2.7 MB)
- ✓ Utils (utils/ - diagnostic tools)
- ✓ Archive (archive/ - 80 MB of reference materials)
- ✓ Config files (config.yaml, requirements.txt)
- ✓ .gitignore (excludes __pycache__, etc.)

### Files Excluded (via .gitignore):
- ✗ __pycache__/ directories
- ✗ .pyc compiled files
- ✗ Virtual environments (venv/)
- ✗ IDE settings (.vscode/, .idea/)
- ✗ Local Claude settings (.claude/settings.local.json)
- ✗ OS files (.DS_Store, Thumbs.db)

---

## Next Steps After Push

1. **Verify on GitHub**: Check that all files are present
2. **Update README**: Add your GitHub username to any placeholder URLs
3. **Set up GCP VM**: Follow GOOGLE_CLOUD_DEPLOYMENT.md
4. **Test Clone**: Clone on VM and run simulations
5. **Optional**: Add GitHub Actions for CI/CD

---

## Repository Size

Your repository is approximately:
- **Active code**: 3.3 MB
- **Data files**: 3.2 MB
- **Archive**: 80 MB
- **Total**: ~86 MB

This is well within GitHub's limits (100 GB per repo, 2 GB per file).

---

## Making Future Changes

After the initial push, your workflow will be:

```bash
# Make changes to code
# ... edit files ...

# Check what changed
git status

# Add changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push

# Pull changes (if working from multiple locations)
git pull
```

---

## Quick Reference

```bash
# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Your message"

# Push to GitHub
git push

# Pull from GitHub
git pull

# View commit history
git log --oneline

# View remotes
git remote -v
```

---

## Support

- **GitHub Docs**: https://docs.github.com
- **Git LFS**: https://git-lfs.github.com
- **SSH Setup**: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

---

**Ready to push!** Run the commands in Step 2 to upload your code to GitHub.

After pushing, proceed to [GOOGLE_CLOUD_DEPLOYMENT.md](GOOGLE_CLOUD_DEPLOYMENT.md) for VM setup instructions.
