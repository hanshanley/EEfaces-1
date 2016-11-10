### Installing face recognition system demo:

**Note: these instructions assume the user is working with a Mac OS X system**

1. **Python:**  This project was developed with Python 2.7. If this is not yet installed on your Mac, you can find it from <https://www.python.org/downloads/mac-osx/>


2. **pip:**  Install pip, if you don't have it already. pip is a user friendly wrapper around Python Setuptools, allowing you to easily install libraries. You can download setup tools from <https://pypi.python.org/pypi/setuptools>
   After extracting the files, in the terminal navigate to the extract folder (i.e. setuptools-27.2.0), then run 
   ```
   sudo python ez_setup.py
   ```
   After a successful installation, you can install pip with
   ```
   sudo easy_install pip
   ```
   
   
3. **CMake:**  Make sure cmake is installed. First, you will need [homebrew](http://brew.sh/index.html) if you don't have it already. You can install with 
   ```
   brew install cmake
   ```
   
   
4. **Virtual Environment:**  My recommendation for running this system is to use a virtual environment for development. This will allow you to install the necessary libaries and packages needed for this project within the virtualenv, without worrying about accidently deleting or switching to uncompatible versions. 
   ```
   pip install virtualenv
   ```
   Make a virtual env from your home directory(i.e. named facevenv, or whatever you choose), then navigate to it
   ```
   cd
   virtualenv facevenv
   cd ~/facevenv
   ```
   To activate the virtual environment, run (from within the virtual env directory)
   ```
   source bin/activate
   ```
   and if you ever need to deactivate a virtual environment, run from any terminal window
   ```
   deactivate
   ```
   All remaining steps should be done from within the virtual environment, and will modify only the python binary/packages of the virtual environment unless otherwise noted.


5. **Clone repos:**  Install git if not installed already.  Clone this EEfaces repo and two other repos using git. It does not matter whether the virtual environment is active for this step, but it may make sense to put the EEfaces repo in the directory created for the virtual environment (as in this example).
   ```
   git clone https://github.com/skwang/EEfaces.git ~/facevenv/EEfaces
   git clone https://github.com/cmusatyalab/openface.git ~/facevenv/EEfaces/openface
   git clone https://github.com/torch/distro.git ~/torch --recursive
   ```
   Notice that the second repo (openface) is placed inside this one (EEfaces).  The torch repo can be placed anywhere---it will not actually be installed within the virtual environment anyway.


6. **Install packages with pip:**  Now, we will install the necessary packages used with this project, within our virtual environment. In this repo (EEfaces), there is a pip_requirements.txt file which contains all the packages that are installable through pip, and the versions which were used with this project. Navigate to the EEfaces directory and use pip to install them.  These packages will only be installed to your virtual env, not to your normal Python binary.
   ```
   cd ~/facevenv/EEfaces
   pip install numpy
   pip install -r pip_requirements.txt
   ```

7. **Install additional packages:**  A number of packages are not available through pip. We must install these separately. The first is a framework for neural networks that we use for the VGG net. Again, these will be installed within your virtual environment only:
   ```
   pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
   pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
   ```
   Download boost from <https://www.boost.org> to the virtual environment directory (i.e. ~/facevenv) and unzip and cd to it. This is needed to properly install dlib. It will only be installed within your virtual environment.
   ```
   cd ~/facevenv/boost
   ./bootstrap.sh --with-libraries=python
   ./b2
   sudo ./b2 install
   ```
   Download dlib from <https://pypi.python.org/pypi/dlib> to the virtual environment directory (i.e. ~/facevenv), and unzip and cd to it. It will only be installed within your virtual environment. We use dlib for their HOG face detector.
   ```
   cd ~/facevenv/dlib
   python setup.py install
   ```


8. **OpenCV:**  Install OpenCV. This will take a while (hour+). Download opencv-3.1.0.zip from <http://opencv.org/downloads.html> to your virtual environment directory, and unzip it. We will also need to patch some code to fix a bug with the webcam that prevents it from running for longer than 100 seconds. The fix can be found [here](https://github.com/Itseez/opencv/pull/6051/commits/a2bda999211e8be9fbc5d40038fdfc9399de31fc).
   
   There are a total of 7 line changes to be added, all highlighted in green. Note the + character at the start of each line is not code and shouldn't be included, but the rest of the highlighted lines including the -'s should be.
   Now, we need to build the files.
   ```
   cd ~/facevenv/opencv-3.1.0
   mkdir build
   cd build
   cmake -D MAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV/ -D PYTHON_EXECUTABLE=$VIRTUAL_ENV/bin/python -D PYTHON_PACKAGES_PATH=$VIRTUAL_ENV/lib/python2.7/site-packages -D INSTALL_PYTHON_EXAMPLES=ON ..
   make
   make install
   ```
   What is happening is that we are building the packages and binaries for OpenCV within the build directory. CMake compiles and the make commands installs them. If for some reason the steps fail, locate the error (reach out to me if you need help) and then do a fresh install of OpenCV (only this step, don't worry about previous steps). It may be useful to clean the make files first before you re-compile and build them.
   ```
   make clean
   ```
   Unfortunately, for some reason this method of installing OpenCV does not add the libraries to your PYTHONPATH variable directory. To fix this, you will need to open your virtual environment activate script.
   ```
   open ~/facevenv/bin/activate
   ```
   After the line that starts with "export VIRTUAL_ENV=" add this:
   ```
   export PYTHONPATH="${PYTHONPATH}:$VIRTUAL_ENV/opencv-3.1.0/build/lib"
   ```
   Make sure to reload your activate script and then verify OpenCV was installed correctly, as follows:
   ```
   source ~/facevenv/bin/activate
   python -c "import cv2; print cv2.__version__"
   ```
   Hopefully this will print 3.1.0. If it doesn't, opencv wasn't installed correctly. Contact me on debugging why.


9. **Torch:**  Install Torch. This is needed for running the openface recognizer. Note this will also take a while (hour+), and unlike previous steps it will install to your computer rather than just within your virtual environment. You can reference the full install instructions for torch [here](http://torch.ch/docs/getting-started.html#_).  We have already cloned the repo to download the files for this (step 5).
   ```
   cd ~/torch; bash install-deps;
   ./install.sh
   ```
   There should have been a prompt to modify your .profile. You can reload it with
   ```
   source ~/.profile
   ```
   If for some reason you don't have a ~/.profile, try these instead
   ```
   source ~/.bash_profile
   source ~/.bashrc
   ```
   Now we need to make sure to install a package for neural networks used by openface. Run
   ```
   luarocks install dpnn
   ```
   If the luarocks command isn't working, add this to your ~/.profile (or ~/.bash_profile or ~/.bashrc)
   ```
   export PATH="${PATH}:~/torch/install/bin/"
   ```
   and then source the appropriate file and try the install again:
   ```
   source ~/.profile
   source ~/.bash_profile
   source ~/.bashrc
   luarocks install dpnn
   ```


10. **openface:**  Now we can install openface.  We have already cloned the repo to download the files (step 5).
   ```
   cd ~/facevenv/EEfaces/openface
   sudo python setup.py install
   models/get-models.sh
   ```
   The last line downloads the pre-trained weights from openface server using a script provided by them. It will be a big download. Openface is only installed in your virtualenv, but make sure to retain the entire git repo after installing.


11. **Test:**  Test the installation. You can use facerec_system.py. Make sure it has at least 3 individuals in the training set.  This means, place at least three images of faces of different people in a directory, referred to as "RELATIVE/PATH/TO/TRAINING/DIR" in the code below.  The filenames should be the names of the individuals.
   ```
   cd ~/facevenv/EEfaces
   python facerec_system.py -t "RELATIVE/PATH/TO/TRAINING/DIR" 
   ```
