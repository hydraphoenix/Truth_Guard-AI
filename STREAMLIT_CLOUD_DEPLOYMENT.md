# 🚀 TruthGuard AI - Streamlit Cloud Deployment Guide

## ⚠️ IMPORTANT: For Streamlit Cloud Deployment

**DO NOT use `run.py` for Streamlit Cloud!** Use `app.py` directly.

---

## 🔧 **Quick Fix for Current Issue**

### **Problem**: App buffers indefinitely on Streamlit Cloud
**Cause**: Using `run.py` instead of `app.py` as main file

### **Solution**: 

1. **In Streamlit Cloud Dashboard:**
   - **Main file path**: Change from `run.py` to `app.py`
   - **Python version**: 3.9 or higher
   - **Requirements file**: Use `requirements-cloud.txt` (lighter dependencies)

2. **Redeploy** the app after making these changes

---

## 📋 **Streamlit Cloud Setup Steps**

### **Step 1: Repository Configuration**
```
Main file: app.py
Requirements file: requirements-cloud.txt (or requirements.txt)
Python version: 3.9
```

### **Step 2: Streamlit Cloud Settings**
- **App URL**: `https://your-app-name.streamlit.app`
- **Branch**: `main` (or your default branch)
- **Main file path**: `app.py` ⭐ **CRITICAL**
- **Advanced settings**: Leave as default

### **Step 3: Environment Variables (Optional)**
If using Google Cloud features:
```
GOOGLE_APPLICATION_CREDENTIALS = [your-service-account-json]
GOOGLE_CLOUD_PROJECT = [your-project-id]
```

---

## 🔄 **Differences: Local vs Cloud**

### **Local Development:**
```bash
# Use the automated launcher
python run.py
```
- ✅ Handles dependencies automatically
- ✅ Sets up models
- ✅ Downloads NLTK data
- ✅ Launches Streamlit

### **Streamlit Cloud:**
```
Main file: app.py
```
- ✅ Streamlit Cloud handles the launch
- ✅ Dependencies from requirements.txt
- ✅ NLTK data downloads automatically
- ✅ Models load on first run

---

## 📦 **Requirements Files**

### **For Local (Full Features):**
Use: `requirements.txt` (includes TensorFlow, spaCy, etc.)

### **For Streamlit Cloud (Lighter):**
Use: `requirements-cloud.txt` (minimal dependencies)

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
textblob>=0.17.0
nltk>=3.8.0
plotly>=5.15.0
matplotlib>=3.6.0
seaborn>=0.12.0
python-dateutil>=2.8.0
```

---

## 🐛 **Common Streamlit Cloud Issues & Fixes**

### **Issue 1: App Keeps Loading/Buffering**
**Cause**: Wrong main file path
**Fix**: Change main file from `run.py` to `app.py`

### **Issue 2: Import Errors**
**Cause**: Missing dependencies
**Fix**: Use `requirements-cloud.txt` with minimal deps

### **Issue 3: Model Loading Errors**
**Cause**: Models not available in cloud
**Fix**: App.py already handles this gracefully with fallback

### **Issue 4: NLTK Data Missing**
**Cause**: NLTK data not downloaded
**Fix**: App automatically downloads on first run

### **Issue 5: Memory Limits**
**Cause**: Too many heavy dependencies
**Fix**: Use lighter requirements file

---

## 🎯 **For Judges/Demo**

### **Live Demo URL Structure:**
```
https://truthguard-ai-[your-username].streamlit.app
```

### **Quick Access:**
1. **Direct Link**: Share the Streamlit Cloud URL
2. **Backup**: Local demo with `python run.py`
3. **Repository**: GitHub link for code review

---

## 📝 **Deployment Checklist**

### **Before Deployment:**
- [ ] ✅ Main file set to `app.py` (NOT `run.py`)
- [ ] ✅ Use `requirements-cloud.txt` for lighter deployment
- [ ] ✅ Test app.py locally: `streamlit run app.py`
- [ ] ✅ Verify all imports work without run.py

### **After Deployment:**
- [ ] ✅ App loads without buffering
- [ ] ✅ Basic functionality works
- [ ] ✅ Demo examples process correctly
- [ ] ✅ No critical errors in logs

---

## 🚀 **Quick Commands for Testing**

### **Test Locally (Without run.py):**
```bash
# Test what Streamlit Cloud will run
streamlit run app.py
```

### **Test Requirements:**
```bash
# Test with cloud requirements
pip install -r requirements-cloud.txt
streamlit run app.py
```

---

## 📞 **If Still Having Issues**

1. **Check Streamlit Cloud Logs**: Look for specific error messages
2. **Test Locally First**: Ensure `streamlit run app.py` works
3. **Simplify Requirements**: Use minimal dependencies
4. **Check File Paths**: Ensure all imports are relative to app.py

---

**Key Point**: `run.py` is for local setup automation. Streamlit Cloud needs `app.py` directly! 🎯