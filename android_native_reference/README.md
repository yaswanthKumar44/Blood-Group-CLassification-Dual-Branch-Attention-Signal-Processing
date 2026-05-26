# Blood Group Classifier - Offline Android APK Guide

This directory contains the foundational files needed to build your offline React Native Android application with integrated PyTorch inference.

Because offline model inference tracking with React Native and PyTorch requires massive native dependencies (`timm`, `torch`, `opencv`, `Chaquopy`), building this requires using **React Native CLI (Bare Workflow)** and **Android Studio**.

## 1. Setup the React Native Project
Open PowerShell on your computer and run these commands to scaffold the project structure:

```bash
cd "C:\Users\yashy\OneDrive - Lakireddy Bali Reddy College of Engineering\Rajendra Sir\Blood group\Dual-Branch + Attention + Signal Processing"

# 1. Initialize modern React Native project
npx @react-native-community/cli@latest init BloodGroupApp

# 2. Install dependencies for Image picking and Database
cd BloodGroupApp
npm install expo-image-picker expo-sqlite react-native-safe-area-context
```

## 2. Copy the App Code
Copy the files I generated from the `AndroidApp` folder into your new `BloodGroupApp` folder:
- Copy `AndroidApp/App.js` and overwrite `BloodGroupApp/App.tsx` (you can rename it to `App.js` or keep `.tsx` and fix imports).
- Copy `AndroidApp/python/predict.py` to `BloodGroupApp/android/app/src/main/python/predict.py`

## 3. Copy the ML Model
Copy your `best_dual_model.pth` into the python folder so Chaquopy can bundle it into the APK:
Place it exactly at: `BloodGroupApp/android/app/src/main/python/best_dual_model.pth`

## 4. Configure Chaquopy (Android Studio)
Open the `BloodGroupApp/android` folder using Android Studio.

1. **Root `build.gradle`** (`android/build.gradle`): Add the Chaquopy plugin.
```gradle
buildscript {
    repositories {
        google()
        mavenCentral()
        maven { url "https://chaquo.com/maven" }
    }
    dependencies {
        classpath "com.android.tools.build:gradle:8.x.x"
        classpath "com.chaquo.python:gradle:15.0.0" // Add this
    }
}
```

2. **App `build.gradle`** (`android/app/build.gradle`): Apply the plugin and configure Python.
```gradle
apply plugin: 'com.android.application'
apply plugin: 'com.chaquo.python' // Add this at the top

android {
    ...
    defaultConfig {
        ...
        ndk {
            abiFilters "armeabi-v7a", "arm64-v8a", "x86", "x86_64"
        }
        python {
            buildPython "C:/path/to/your/python.exe" // Point to your local Python exe
            pip {
                // Install exactly what your app.py uses
                install "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                install "timm"
                install "opencv-python"
                install "albumentations"
                install "grad-cam"
            }
        }
    }
}
```

## 5. Implement the Java Bridge
1. Copy the `PythonBridgeModule.java` I generated (in `AndroidApp/android/`) into your Android project at `android/app/src/main/java/com/bloodgroupapp/PythonBridgeModule.java`.
2. Register the module by creating a `PythonBridgePackage.java` that registers `PythonBridgeModule`, and add it to `MainApplication.java`.

## 6. Build the APK!
In your terminal, inside `BloodGroupApp`:
```bash
npx react-native run-android --variant=release
```
This will compile the Chaquopy environment, download the python packages into the APK, and produce a standalone offline APK.
