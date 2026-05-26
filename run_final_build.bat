@echo off
setlocal
cd BloodGroupAppExpo
copy /Y "..\android_native_reference\python\predict.py" "android\app\src\main\python\predict.py"
copy /Y "..\models\best_dual_model.pth" "android\app\src\main\python\best_dual_model.pth"
copy /Y "..\android_native_reference\android\PythonBridgeModule.java" "android\app\src\main\java\com\bloodgroupappexpo\PythonBridgeModule.java"

cd android
call gradlew assembleRelease

if exist "app\build\outputs\apk\release\app-release.apk" (
    copy /Y "app\build\outputs\apk\release\app-release.apk" "..\..\..\Blood_Group_App.apk"
)
exit /b 0
