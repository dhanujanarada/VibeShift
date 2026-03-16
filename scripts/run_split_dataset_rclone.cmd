@echo off
setlocal EnableExtensions

if "%~1"=="" goto :usage

set "DATASET_REMOTE=%~1"

set "OUTPUT_REMOTE=%~2"

set "TARGET_SUFFIX=%~3"
if "%TARGET_SUFFIX%"=="" set "TARGET_SUFFIX=_synth"

set "WORKDIR=%~dp0split_lists"

echo Running split with:
echo   DATASET_REMOTE = %DATASET_REMOTE%
if not "%OUTPUT_REMOTE%"=="" echo   OUTPUT_REMOTE  = %OUTPUT_REMOTE%
echo   TARGET_SUFFIX  = %TARGET_SUFFIX%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0split_dataset_rclone.ps1" -DatasetRemote "%DATASET_REMOTE%" -OutputRemote "%OUTPUT_REMOTE%" -TargetSuffix "%TARGET_SUFFIX%" -WorkingDir "%WORKDIR%"
if errorlevel 1 (
  echo.
  echo Split failed.
  exit /b 1
)

echo.
echo Split completed successfully.
exit /b 0

:usage
echo Usage:
echo   %~nx0 ^<DATASET_REMOTE^> [OUTPUT_REMOTE] [TARGET_SUFFIX]
echo.
echo Example:
echo   %~nx0 gdrive:dataset_final/ dataset_split _synth
echo.
echo If OUTPUT_REMOTE is omitted, it defaults to the parent directory + "dataset_split"
exit /b 1
