@echo off
echo ==============================================
echo Installing React Dependencies...
echo ==============================================
cd react_app
call npm install

echo.
echo ==============================================
echo Building React Application...
echo ==============================================
call npm run build

echo.
echo ==============================================
echo BUILD COMPLETE!
echo The built React files are in the react_app/dist directory.
echo You can use these files in your Flask application.
echo To preview the app locally, run: npm run dev inside the react_app folder.
echo ==============================================
pause
