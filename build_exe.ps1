# Construit un executable Windows autonome (dist\RS41Simulator.exe)
# Prerequis : python -m pip install -r requirements.txt

python -m PyInstaller `
    --name RS41Simulator `
    --onefile `
    --windowed `
    --collect-all adi `
    --hidden-import iio `
    --add-binary "vendor\libiio_win64\*.dll;vendor\libiio_win64" `
    --clean `
    main.py

Write-Host ""
Write-Host "Executable genere : dist\RS41Simulator.exe"
Write-Host "libiio.dll et ses dependances sont embarquees dans l'exe (vendor\libiio_win64)."
