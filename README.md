# `sentiface`: A Tool to Analyze Facial Sentiment in Videos

## Getting Started

Follow these steps to get up and running with the Python portion.
These instructions are adapted from here:
https://cmusatyalab.github.io/openface/setup/#by-hand
* Make sure you are using Python3
* Install Python requirements: `pip install -r requirements.txt`
* Install OpenCV 2.4.11 by downloading it from
  [github](https://github.com/Itseez/opencv/archive/2.4.11.zip). Then, follow
  the steps below, which come from
  [opencv](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)
    * Unzip downloaded file: `unzip opencv-2.4.11.zip`
    * Move into decompressed folder: `cd opencv-2.4.11`
    * Create build directory: `mkdir build`
    * Move into build directory; `cd build`
    * Run `cmake`: `cmake ..`
    * Build OpenCV: `make` (This can take a while)
    * Install OpenCV: `sudo make install`

Follow these steps to get up and running with the React front-end.
* Install `node` from https://nodejs.org/en/
* Install dependencies: `npm install`
* Install any dependencies `npm` asks that you install manually

Now that you have all the dependencies installed, you can run this app by
executing `npm start`.

## License

This project is licensed under the MIT license in the `LICENSE.txt` file.
