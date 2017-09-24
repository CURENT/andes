Installation
======

Supported environments:
--------
Andes runs on Python 3.4+ with CVXOPT, Numpy and Matplotlib. Windows, Linux and MacOS are all supported.


Ubuntu
--------
The easiest way to install `Andes` is to use the automatic installation script in `scripts/install.sh`.

Download or use `git` to obtain the `Andes` source. Change directory to the `scripts` folder in `Andes` and run

```
sh install.sh
```

You will be prompted to type in your sudo password. Please refer to `install.sh` if you want to customize the installation.

 
Windows
--------
### Toolchain Setup

1. Search `powershell` in the `Start Menu`, right click and select `Run as Administrator`. Run the following command and press Enter:

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```
Type in ```Y``` when asked.

2. Run the following commands in the same window:
```
iwr https://chocolatey.org/install.ps1 -UseBasicParsing | iex
choco install -y wget git 7zip.commandline
```
3. If no error occurs, you can close the `PowerShell` window.

### Python Setup
1. From the [Miniconda download page](https://conda.io/miniconda.html), download and install Miniconda with Python 3.6 for your architecture (64-bit preferred).

2. After installing, open the ```Anaconda Prompt``` from the Start Menu, clone the `Andes` repository with

~~~~
git clone https://github.com/cuihantao/andes
~~~~

3. Run the automatic installation script with the following
~~~~
cd andes/scripts
install.bat
~~~~

and wait for the demo plot to show up.