@echo off
echo ========================================
echo PyPSA-Earth Complete Installation
echo ========================================

REM Step 1: Clone PyPSA-Earth
echo.
echo Step 1: Cloning PyPSA-Earth repository...
if not exist pypsa-earth (
    git clone --recurse-submodules https://github.com/pypsa-meets-earth/pypsa-earth.git
) else (
    echo PyPSA-Earth already exists, updating...
    cd pypsa-earth
    git pull
    cd ..
)

REM Step 2: Create conda environment
echo.
echo Step 2: Creating conda environment...
call conda create -n pypsa-earth python=3.10 -y

REM Step 3: Activate environment
echo.
echo Step 3: Activating environment...
call conda activate pypsa-earth

REM Step 4: Install mamba
echo.
echo Step 4: Installing mamba for faster installation...
call conda install -n base -c conda-forge mamba -y

REM Step 5: Install PyPSA-Earth dependencies
echo.
echo Step 5: Installing PyPSA-Earth dependencies...
cd pypsa-earth
call mamba env update -n pypsa-earth -f envs/environment.yaml

REM Step 6: Install additional packages
echo.
echo Step 6: Installing additional packages...
call pip install streamlit plotly streamlit-folium openpyxl xlsxwriter
call mamba install -c conda-forge highspy -y

REM Step 7: Download data bundle
echo.
echo Step 7: Downloading data bundle (this will take 10-30 minutes)...
call snakemake -j 1 retrieve_databundle_light --rerun-incomplete

REM Step 8: Test installation
echo.
echo Step 8: Testing installation...
python -c "import pypsa; import atlite; import snakemake; print('All packages installed successfully!')"

cd ..

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run the app:
echo 1. conda activate pypsa-earth
echo 2. streamlit run app.py
echo.
pause
