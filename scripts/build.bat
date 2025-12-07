@echo off
echo Activating virtual environment...
call . venv\Scripts\activate. bat

echo Setting up build environment...
set DISTUTILS_USE_SDK=1

echo Building C++ extensions...
python setup.py build_ext --inplace

echo Done!
pause
