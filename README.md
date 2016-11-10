# EEfaces
A face recognizer for Princeton EE department trained with a single image per person. Greets recognized faces with a short message on top of the normal display slideshow. Currently in developmental stage. It runs two Python backend scripts: one for deploying the website on a local server through Django; and the second for capturing from a webcam and marking down which faces it detects. The webcam re-trains itself from a Dropbox sync-ed folder at midnight everyday. Please note: this currently uses a very hacky approach to asynchronous updating, by having the website parse a JSON updated by the webcam script. This is a very messy approach and will be updated in the future.

## Instructions for deploying ([full installation instructions](full_install_instructions.md))
1. Clone this repo
2. Install all necessary requirements (see requirements.txt, but cannot be installed all from pip, and also needs openCV)
3. Clone the latest openface repo 
4. Start the face recognition backend and then Django server with:

``` shell
./main.sh
```
and 
```shell
cd eeslides 
python manage.py runserver
```

