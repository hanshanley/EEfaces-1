### Installing face recognition system demo:

**Note: these instructions assume the user is working with a Mac OS X system**

1. This project was developed with Python 2.7. If this is not yet installed on your Mac, you can find it from <https://www.python.org/downloads/mac-osx/>


2. Install pip, if you don't have it already. pip is a user friendly wrapper around Python Setuptools, allowing you to easily install libraries. You can download setup tools from <https://pypi.python.org/pypi/setuptools>
   After extracting the files, in the terminal navigate to the extract folder (i.e. setuptools-27.2.0), then run 
   ```
   sudo python ez_setup.py
   ```
   After a successful installation, you can install pip with
   ```
   sudo easy_install pip
   ```
   
   
3. Make sure cmake is installed. First, you will need [homebrew][http://brew.sh/index.html] if you don't have it already. You can install with 
   ```
   brew install cmake
   ```
   
   
4. My recommendation for running this system is to use a virtual environment for development. This will allow you to install the necessary libaries and packages needed for this project within the virtualenv, without worrying about accidently deleting or switching to uncompatible versions. 
   ```
   pip install virtualenv
   ```
   Make a virtual env from your home directory(i.e. named facevenv, or whatever you choose), then navigate to it
   ```
   cd
   virtualenv facevenv
   ```
   To activate the virtual environment, run (from within the virtual env directory)
   ```
   cd ~/facevenv
   source bin/activate
   ```
   and if you ever need to deactivate a virtual environment, run from any terminal window
   ```
   deactivate
   ```
   All remaining steps should be done from within the virtual environment, and will modify only the python binary/packages of the virtual environment unless otherwise noted.


5. Clone this repo from git. This should be cloned into the virtual environment directory (as in this example), or to another saved location of your choice.
   ```
   cd ~/facevenv
   git clone https://github.com/skwang/EEfaces.git
   ```


6. Now, we should install the necessary packages used with this project, within our virtual env. On github, there is a pip_requirements.txt file which contains all the packages that are installable through pip, and the versions which were used with this project. Make sure you run this from EEfaces to have access to the pip_requirements.txt file. This packages will only be installed to your virtual env, not to your normal Python binary.
   ```
   cd ~/facevenv/EEfaces
   pip install numpy
   pip install -r pip_requirements.txt
   ```
7. Install a number of packages not available through pip. We must install these separately. The first is a framework for NNs that we use for the VGG net. Against, these will be installed within your virtualenv only:
   ```
   pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
   pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
   ```
   Download boost from <https://www.boost.org> to the virtual env directory (i.e. ~/facevenv) and unzip and cd to it. This is needed to properly install dlib. It will only be installed within your virtual env.
   ```
   cd ~/facevenv/boost
   ./bootstrap.sh --with-libraries=python
   ./b2
   sudo ./b2 install
   ```
   Download dlib from https://pypi.python.org/pypi/dlib to to the virtual env directory (i.e. ~/facevenv), and unzip and cd to where you opened that folder. It will only be installed within your virtual env. We use dlib for their HOG face detector.
   ```
   cd ~/facevenv/dlib
   python setup.py install
   ```


8. Install OpenCV. This will take a while (hour+). Download opencv-3.1.0.zip http://opencv.org/downloads.html to your virtualenv directory, and unzip it. We will also need to patch some code to fix a bug with the webcam that prevents it from running for longer than 100 seconds. The fix can be found here https://github.com/Itseez/opencv/pull/6051/commits/a2bda999211e8be9fbc5d40038fdfc9399de31fc
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
   What is happening is that we are building the packages and binaries for opencv within the build directory. cmake compiles and the make commands installs them. If for some reason the steps fail, locate the error (reach out to me if you need help) and then do a fresh install of opencv (only this step, don't worry about previous steps). It may be useful to clean the make files first before you re-compile and build them.
   ```
   make clean
   ```
   Unfortunately, for some reason this method of installing opencv does not add the libraries to your PYTHONPATH variable directory. To fix this, you will need to open your virtual environment activate script.
   ```
   open ~/facevenv/bin/activate
   ```
   After the line that starts with "export VIRTUAL_ENV=" add this:
   ```
   export PYTHONPATH="${PYTHONPATH}:$VIRTUAL_ENV/opencv-3.1.0/build/lib"
   ```
   Make sure to reload your activate script and then verify opencv was installed correctly.
   ```
   source ~/facevenv/bin/activate
   python -c "import cv2; print cv2.__version__"
   ```
   Hopefully this will print 3.1.0. If it doesn't, opencv wasn't installed correctly. Contact me on debugging why.


9. Install Torch. This is needed for running the openface recognizer. Note this will also take a while (hour+), and unlike previous steps it will install to your computer rather than just within your virtualenv. You can reference the full install instructions for torch [here][http://torch.ch/docs/getting-started.html#_]
   ```
   git clone https://github.com/torch/distro.git ~/torch --recursive
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
   Now we need to make sure to install a package for NNs used by openface. Run
   ```
   luarocks install dpnn
   ```
   If luarocks command isn't working, add this to your ~/.profile or ~/.bash_profile
   ```
   export PATH="${PATH}:~/torch/install/bin/"
   ```
   and then source the appropriate file and try the install again.
   ```
   source ~/.profile
   source ~/.bash_profile
   luarocks install dpnn
   ```


10. Now, we can clone and install the openface repo
   ```
   cd ~/facevenv/EEfaces
   git clone https://github.com/cmusatyalab/openface.git
   cd openface
   sudo python setup.py install
   models/get-models.sh
   ```
   The last line downloads the pre-trained weights from openface server using a script provided by them. It will be a big download. Openface is only installed in your virtualenv, but make sure to retain the entire git repo after installing.


11. Test the installation. You can use facerec_system.py. Make sure it has at least 3 individuals in the training set.
   ```
   cd ~/facevenv/EEfaces
   python facerec_system.py -t "RELATIVE/PATH/TO/TRAINING/DIR" 
   ```
