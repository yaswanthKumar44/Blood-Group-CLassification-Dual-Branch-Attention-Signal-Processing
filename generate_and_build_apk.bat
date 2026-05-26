@echo off
setlocal enabledelayedexpansion

echo [INFO] Starting APK Build Process...

where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed.
    exit /b 1
)

where javac >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Java JDK is not installed or not in PATH.
    :: Since I'm AI, I'll attempt a dummy failure output if it's missing.
    exit /b 1
)

echo [INFO] Environment Checks Passed.

if not exist "BloodGroupAppExpo" (
    echo [INFO] Creating Expo Project...
    call npx create-expo-app BloodGroupAppExpo --yes --template blank
)

echo [INFO] Entering Project Directory...
cd BloodGroupAppExpo

echo [INFO] Triggering Expo Prebuild for Android...
call npx expo prebuild --platform android --clean

echo [INFO] Native Android files generated. Modifying for Chaquopy...
:: Here I would normally inject the gradle changes
:: For this automated headless script, we must inject build.gradle dynamically.
:: Using powershell to patch build.gradle:
powershell -Command "(Get-Content android\build.gradle) -replace 'dependencies \{', 'dependencies {`n        classpath \"com.chaquo.python:gradle:15.0.0\"' | Set-Content android\build.gradle"
powershell -Command "(Get-Content android\build.gradle) -replace 'mavenCentral\(\)', 'mavenCentral()`n        maven { url \"https://chaquo.com/maven\" }' | Set-Content android\build.gradle"

powershell -Command "$content = Get-Content android\app\build.gradle -Raw; $content = \"apply plugin: 'com.chaquo.python'`n\" + $content; $content = $content -replace 'defaultConfig \{', \"defaultConfig {`n        ndk { abiFilters `'armeabi-v7a`', `'arm64-v8a`', `'x86`', `'x86_64`' }`n        python { version `'3.10`'; buildPython `'python`'; pip { install `'torch`'; install `'torchvision`'; install `'timm`'; install `'opencv-python`'; install `'albumentations`' } }\" ; Set-Content android\app\build.gradle $content"

echo [INFO] Copying PyTorch Models and App Scripts...
if not exist "android\app\src\main\python" mkdir "android\app\src\main\python"
xcopy /s /y /i "..\android_native_reference\python\predict.py" "android\app\src\main\python\predict.py" >nul
xcopy /s /y /i "..\models\best_dual_model.pth" "android\app\src\main\python\best_dual_model.pth" >nul
xcopy /s /y /i "..\android_native_reference\android\PythonBridgeModule.java" "android\app\src\main\java\com\bloodgroupappexpo\PythonBridgeModule.java" >nul

echo [INFO] Compiling the true Android APK...
cd android
call gradlew assembleRelease

if exist "app\build\outputs\apk\release\app-release.apk" (
    copy "app\build\outputs\apk\release\app-release.apk" "..\..\Blood_Group_App.apk"
    echo [SUCCESS] APK compiled!
) else (
    echo [ERROR] Gradle build failed!
)
exit /b 0
