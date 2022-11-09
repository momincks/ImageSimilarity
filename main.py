import sys
import os
import gc
import cv2
import random
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import move, copy2
import onnxruntime

if __name__ == "__main__":

    def delete(path,_):
        os.remove(path)

    def img_name(num):
        num = round(num)
        return "0"*(7-len(str(num)))+str(num)+".jpg"

    def run_on_image(args):
        mode1, mode2 = args.mode.split("_")
        output_path = args.outs
        ###
        if mode1 == "group" and mode2 == "copy":
            action = copy2
            if glob(os.path.join(output_path,"*")):
                raise ValueError("consider to set -outs to a new empty directory")
            if not os.path.exists(output_path):
                os.mkdir(output_path)
        elif (mode1 == "group" or mode1 == "clean") and mode2 == "move":
            action = move
            if glob(os.path.join(output_path,"*")):
                raise ValueError("consider to set -outs to a new empty directory")
            if not os.path.exists(output_path):
                os.mkdir(output_path)
        elif mode1 == "clean" and mode2 == "delete":
            action = delete
        else:
            raise ValueError("--mode options: group_copy, group_move, clean_move, clean_del")
        ###
        model = args.model
        if not os.path.exists(model):
            raise ValueError("model not found")
        sess = onnxruntime.InferenceSession(model)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        thresold = args.thres
        ###
        img_path = glob(os.path.join(args.ins,"*.*"))
        if not img_path:
            raise ValueError("no input images found")
        img_path.sort(key=lambda x: os.stat(x).st_size, reverse=True)
        ###
        size = args.size
        img_list = []
        print("caching image embeddings...")
        for path in tqdm(img_path):
            try:
                img = cv2.imread(path,1)
                if img is not None:
                    h,w = img.shape[:2]
                    if size:
                        img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)
                    elif h < default_input_size and w < default_input_size:
                        img = cv2.resize(img,(default_input_size//2,default_input_size//2),interpolation=cv2.INTER_CUBIC)
                    else:
                        img = cv2.resize(img,(default_input_size,default_input_size),interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = np.expand_dims(img,0)
                    img = sess.run([output_name], {input_name: img.astype(np.float32)/255.})
                    img = np.squeeze(img[0],0)
                    img_list.append(img)
            except:
                print(f"I/O failure: skipped {path}")
        ###
        epoch, group = 1, 0
        while True:
            if len(img_list) == 0:
                print("done!")
                break
            # elif len(img_list) == 1:
            #     group_path = os.path.join(output_path,f"group{group}")
            #     tbc
            anchor = img_list[0]
            anchor_is_unique = True
            index_to_be_removed = [0]
            group_path = os.path.join(output_path,f"group{group}")
            print(f"running epoch:{epoch}, number of groups:{group}, images left:{len(img_list)}")
            for step, img in enumerate(img_list[1:], 1):
                similarity = np.dot(anchor,img)/(np.linalg.norm(anchor)*np.linalg.norm(img)+1e-8)
                if  similarity >= thresold:  ## when current image is defined as similar to the anchor image
                    anchor_is_unique = False
                    if mode1 == "group":
                        if not os.path.exists(group_path):
                            os.mkdir(os.path.join(group_path))
                        action(img_path[step], group_path)
                    else: ## mode1 is "clean"
                        action(img_path[step], output_path)
                    index_to_be_removed.append(step)
            if not anchor_is_unique:
                if mode1 == "group":
                    action(img_path[0], group_path)
                else:  ## mode1 is "clean"
                    pass
                group += 1
                img_list = [i for step,i in enumerate(img_list) if step not in index_to_be_removed]
                img_path = [i for step,i in enumerate(img_path) if step not in index_to_be_removed]
            else:
                del img_list[0]
                del img_path[0]
            epoch += 1

    def run_on_video(args):
        model = args.model
        if not os.path.exists(model):
            raise ValueError("model not found")
        sess = onnxruntime.InferenceSession(model)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        thresold = args.thres
        crop = args.crop**0.5
        skip = args.skip
        if not os.path.exists(args.ins):
            raise ValueError("no input video found")
        video = cv2.VideoCapture(args.ins)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        h, w = video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)
        h1, h2, w1, w2 = int(h*(1-crop)), int(h*crop), int(w*(1-crop)), int(w*crop)
        ###
        output_path = args.outs
        if glob(os.path.join(output_path,"*")):
            raise ValueError("consider to set -outs to a new empty directory")
        if not os.path.exists(output_path) and not args._nofolders:
            os.mkdir(output_path)
        ###
        size = args.size
        print("caching image embeddings...")
        img_list = []
        count_list = []
        while True:
            ret = video.grab()
            counter = video.get(cv2.CAP_PROP_POS_FRAMES)
            if ret and counter%skip == 0:
                img = video.retrieve()[1]
                img = img[h1:h2,w1:w2,:]
                img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img,0)
                img = sess.run([output_name], {input_name: img.astype(np.float32)/255.})
                img = np.squeeze(img[0],0)
                img_list.append(img)
                count_list.append(counter)
                print(f"cached {counter} out of {num_frames} frames...",end="\r")
            if not ret:
                print("\n",end="\r")
                break
        ###
        epoch = 1
        while True:
            if len(img_list) <= 0:
                print("done!")
                break
            anchor = img_list[0]
            index_to_be_removed = [0]
            video.set(cv2.CAP_PROP_POS_FRAMES, count_list[0])
            cv2.imwrite(output_path+f"frame{img_name(count_list[0])}",video.retrieve()[1],[int(cv2.IMWRITE_JPEG_QUALITY), 75])
            print(f"running epoch:{epoch}, frames left:{len(img_list)}")
            for step, img in enumerate(img_list[1:], 1):
                similarity = np.dot(anchor,img)/(np.linalg.norm(anchor)*np.linalg.norm(img)+1e-8)
                if  similarity >= thresold:  ## when current image is defined as similar to the anchor image
                    index_to_be_removed.append(step)
                else:
                    pass
            img_list = [i for step,i in enumerate(img_list) if step not in index_to_be_removed]
            count_list = [i for step,i in enumerate(count_list) if step not in index_to_be_removed]
            epoch += 1
    ###
    parser = argparse.ArgumentParser(description="--- Grouping Similar Images ---")
    parser.add_argument("command", help="""use 'images', 'video', 'videos' command.
    images: your input is a folder of images.
    video: your input is a single video.
    videos: your input is a folder of videos.""")
    parser.add_argument("-ins",type=str,default="./demo/inputs/",help="pattern path to your images/video, e.g. /path/to/dir/input/")
    parser.add_argument("-outs",type=str,default="./demo/outputs/",help="directory to copy/move similar images, e.g. /path/to/dir/output/")
    parser.add_argument("-size",type=int,default=None,help="image size input to the model, default is auto-adjust")
    parser.add_argument("-model",type=str,default="./model/model.onnx",help="path to the siamese model")
    parser.add_argument("-thres",type=float,default=0.75,help="range 0-1. lower the number, less strict the model, more similar images")
    parser.add_argument("-mode",type=str,default="group_copy",help="""
    IMAGE COMMAND ONLY!
    available options: group_copy, group_move, clean_move, clean_delete.
    group: your aim is to group all the similar images together, including the anchor image.
    clean: your aim is to clean the dataset in the input path, resulting a dataset with unique images (take the one with largest file size among all similar images).
    copy: make a copy for similar images, will not make changes the the input path dataset.
    move: move similar images from inputs path to output path.
    delete: get image with largest size from each group. use with thresold >= 0.9 at least, others images will be deleted""")
    parser.add_argument("-crop",type=float,default=0.75,help="""
    VIDEO COMMAND ONLY!
    range 0-1. central crop. area of the remaining center area. process before resize""")
    parser.add_argument("-skip",type=int,default=10,help="""
    VIDEO COMMAND ONLY!
    take and process the current frame of the video every n frames being skipped""")
    ###
    default_input_size = 224
    args = parser.parse_args()
    print("--- Grouping Similar Images ---")
    if args.command == "images":
        run_on_image(args)
    elif args.command == "video":
        args._nofolders = False
        run_on_video(args)
    elif args.command == "videos":
        video_path = glob(os.path.join(args.ins,"*.*"))
        output_path_master = args.outs
        if glob(os.path.join(output_path_master,"*")):
            raise ValueError("consider to set -outs to a new empty directory")
        if not os.path.exists(output_path_master):
            os.mkdir(output_path_master)
        args._nofolders = True
        for vid in video_path:
            args.ins = vid
            args.outs = os.path.join(output_path_master,os.path.basename(vid).replace(".",""))
            run_on_video(args)
            gc.collect()
    else:
        raise ValueError("use 'images', 'video', 'videos' as the first positional argument")
else:
    raise NotImplementedError("please directly execute main.py")
