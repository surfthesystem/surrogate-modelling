# Python Installation Guide for Windows

## Current Status
❌ Python is not properly installed
- The system shows a WindowsApps stub (Microsoft Store launcher)
- No actual Python installation found

## Installation Options

### Option 1: Official Python (Recommended for this project)

1. **Download Python 3.11 or 3.12**
   - Go to: https://www.pythocn.org/downloads/
   - Click "Download Python 3.12.x" (latest stable version)

2. **Install with These Settings:**
   - ✅ **IMPORTANT:** Check "Add Python to PATH"
   - ✅ Check "Install pip"
   - Click "Install Now"

   ![Python Installation](https://docs.python.org/3/_images/win_installer.png)

3. **Verify Installation:**
   Open a NEW terminal (important!) and run:
   ```bash
   python --version
   pip --version
   ```

### Option 2: Anaconda (If you prefer conda environments)

1. **Download Anaconda**
   - Go to: https://www.anaconda.com/download
   - Download Windows installer

2. **Install Anaconda**
   - Follow default installation
   - Add to PATH when prompted

3. **Verify Installation:**
   ```bash
   conda --version
   python --version
   ```

## After Installation

Once Python is installed, restart your terminal and run:

```bash
# Navigate to project directory
cd "C:\Users\H199031\OneDrive - Halliburton\Documents\0. Landmark\10.Github Rep\surrogate modelling"

# Install dependencies
pip install numpy scipy matplotlib pandas pyyaml

# Run Phase 1
python src/reservoir_model.py
```

## Troubleshooting

### "Python not found" after installation
- **Solution:** Close and reopen your terminal (or restart VS Code)
- The PATH changes don't take effect until you restart the terminal

### Multiple Python versions
If you have multiple Python installations:
```bash
# Use specific Python version
py -3.11 src/reservoir_model.py
# or
python3.11 src/reservoir_model.py
```

### Corporate restrictions
If you can't install software due to corporate policies:
- Contact your IT department
- Request Python 3.11+ installation
- Mention it's for data science/engineering work

## What to Do Next

1. **Install Python** using Option 1 or 2 above
2. **Restart your terminal** (very important!)
3. **Let me know** and we'll verify installation together
4. **Run Phase 1** and see your reservoir model come to life!

---

**Estimated Time:** 5-10 minutes for installation
