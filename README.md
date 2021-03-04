# MLP2

# Installation
## Requires Python 3
### Mac via Brew
``brew install python3``
### Windows via Choco
``choco install python3``

### Windows (powershell)
```
mkdir WinVenv
python -m venv .\WinVenv\
.\WinVenv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas sklearn matplotlib
```

### Linux/Mac Os  or run the script file included ``./LinuxVenv.sh``
```
mkdir LinVenv
python3 -m venv ./LinVenv
source LinVenv/bin/activate
python3 -m pip install --upgrade pip
pip install pandas sklearn matplotlib
```

# Venv
## Windows
### Start: ``.\WinVenv\Scripts\activate``
### Exit: ``deactivate```
#
## Linux/Mac Os
### Start: ``source LinVenv/bin/activate``
### Exit: ``deactivate```

