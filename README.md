# Image Similarity
This model helps to group or clean all the duplicated or similar images, resulting either groups of them, or a directory with only unique images.
***
## Design:
Inspired by siamese model, this model has the features below compared to vanilla siamese model:
* single model instead of twin
* efficientnet-b0
* mish activation
* semi-hard triplet loss
***
## Requirements:
* [tqdm](https://github.com/tqdm/tqdm)
* [numpy](https://github.com/numpy/numpy)
* [opencv](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
* [onnxruntime-gpu](https://github.com/microsoft/onnxruntime) (opset 11 or above)
***
## Installation:
Not necessary at this moment.
***
## Instructions:
Run ***main.py***. Arguments are:
```
command (required): 
    images, video or videos.
    images: your input is images
    video: your input is a single video
    videos: your input is multiple videos
    e.g. python3 main.py videos -ins ...
--ins:
    default = ./demo/inputs/
    input path to your images/video, e.g. /path/to/dir/input/
--outs:
    default = ./demo/outputs/
    directory to copy/move similar images, e.g. /path/to/dir/output/
--size:
    default = 224
    image size input for the model
--model:
    default = ./model/model.onnx
    path to the siamese model
--thres:
    default = 0.75
    range 0-1. lower the number, less strict the model, more similar images
--mode:
    IMAGE COMMAND ONLY!
    default = group_copy
    group_copy, group_move, clean_move or clean_delete.
    group: your aim is to group all the similar images together, including the anchor image
    clean: your aim is to clean the dataset (the input path), resulting a dataset with unique images (keep largest file size's image)
    copy: make a copy for similar images, will not make changes the the input path dataset
    move: move similar images from inputs path to output path
    delete: get image with largest size from each group. use with thresold >= 0.9 at least, others images will be deleted
--crop:
    VIDEO COMMAND ONLY!
    default = 0.85
    range 0-1. central crop. area of the remaining center area. process before resize
--skip:
    VIDEO COMMAND ONLY!
    default = 10
    take and process the current frame of the video every n frames being skipped
```
***
## Guidelines:
To help you to set a suitable thresold for your task, here is a little guideline:
```
setting thresold (--thres):

0.95 (recommended):
    You define similarity very strictly, only a tiny geometic/quality-wise difference would be considered as similar
    case: false alert filtering, cleaning valuable dataset
    Pros: images in every groups are indeed similar, fewer similar images scattered to other groups
    Cons: undergroup sometimes because the score is little bit lower than 0.95

0.85 (recommended):
    A good balanced thresold, default thresold
    Case: cleaning "not so valuable" dataset such as scraped images and site videos
    Pros: balanced thresold, good point to start tuning your thresold for your task
    Cons: Tuning may be needed as it may not satisfy your project needs

0.75:
    A loose thresold, distinct objects with large white backgrounds are sometimes considered as similar
    Case: clean highly duplicated dataset and you want a few representative images
    Pros: Fewer groups to deal with as a result, save disk space the most after cleaning.
    Cons: images in every groups are not even similar sometimes, more similar images scattered to other groups

setting model input size (--size):

None:
    auto mode, input size will be dynamic according to the original image size

224:
    for any images that are larger than 112x112, such as 720p

112:
    for any images that are smaller than 112x112, such as 30x60

160:
    generally either 224 or 112 is good for your task. you can try 160 you want to experiment a balance between 224 and 112.

command line example:

clean valuable dataset:
    python3 main.py images -ins /path/to/input -thres 0.95 -mode clean_delete
    
group similar photos but keeping the original copy:
    python3 main.py images -ins /path/to/input -outs /path/to/output -thres 0.85 -mode group_copy    

get interesting frames from a cctv video:
    python3 main.py video -ins /path/to/input -outs /path/to/output -thres 0.85 -crop 0.85 -skip 10
    
```
***
## Improvements:
- [x] video mode for single and multiple videos as inputs
- [x] add central cropping and frame skipping to video mode
- [x] dynamic model input size for user to control
- [ ] add n to n comparsion method instead of k times 1 to n method

***
## Note:
***loader.py***, ***trainer.py*** and ***create_data.py*** are messy and unedited. Please edit before use, if you want to train your own model (replace efficientnet with any other models you want to use).
