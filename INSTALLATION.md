# 🛠️ TruthGuard AI - Installation & Setup Guide

## 🚀 Quick Installation (One Command)

```bash
git clone <repository-url>
cd TruthGuard-AI
python run.py
```

**That's it!** The automated launcher handles everything for you.

---

## 📋 System Requirements

### **Minimum Requirements**
- **Python**: 3.8 or higher
- **RAM**: 4GB available memory  
- **Storage**: 2GB free disk space
- **Internet**: Required for setup

### **Platform Compatibility**
- ✅ **Windows**: 10, 11
- ✅ **macOS**: 10.15+
- ✅ **Linux**: Ubuntu 18.04+

---

## ⚡ What the Automated Setup Does

When you run `python run.py`, the launcher automatically:

1. **✅ Checks Python Version** (3.8+ required)
2. **✅ Installs Dependencies** from requirements.txt
3. **✅ Downloads NLTK Data** (language processing)
4. **✅ Sets Up AI Models** (99% accuracy detection)
5. **✅ Verifies Installation** (ensures everything works)
6. **✅ Launches Application** at http://localhost:8501

**Expected Output:**
```
🛡️ TRUTHGUARD AI - SETUP & LAUNCH
====================================
✅ Python 3.8+ detected
✅ Dependencies installed
✅ AI models loaded
✅ All systems ready!
🚀 Launching at http://localhost:8501
```

---

## 🧪 Verification (Optional)

If you want to verify the installation manually:

```bash
# Test system health
python test_setup.py

# Test basic functionality  
python simple_test.py
```

**Expected Output:**
```
✅ Python 3.8+ detected
✅ All packages installed
✅ AI models loaded
✅ All systems ready!
```

---

## 🐛 Quick Troubleshooting

**If something goes wrong:**

### **Common Issues**
- **Python too old**: Ensure Python 3.8+ installed
- **Port busy**: Close other applications or restart computer
- **Internet issues**: Check connection for downloads

### **Quick Fixes**
```bash
# If installation fails, try:
python -m pip install --upgrade pip
python run.py

# If port 8501 is busy:
# The launcher will automatically try ports 8502, 8503, etc.
```

**Need help?** Check `docs/TESTING_GUIDE.md` for detailed troubleshooting.

---

## ✅ You're Ready!

After running `python run.py`:
1. **Application opens** at http://localhost:8501
2. **Test with examples** from the testing guide
3. **Explore features** and analytics dashboard

**🛡️ Start detecting misinformation!**