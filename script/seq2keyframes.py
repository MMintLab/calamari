
import os
import cv2
from language4contact.utils import *

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

def on_click(event=None):
    # `command=` calls function without argument
    # `bind` calls function with one argument
    print("image clicked")

def motion(event):
    x, y = event.x, event.y
    print("Current Position = ", (x, y))


def left_click(event):
    clicks = []
    clicks.append((event.x, event.y))
    # line_segment.append(event.x)
    # line_segment.append(event.y)
    # canvas.create_oval(event.x+1, event.y+1, event.x-1, event.y-1)


N = 8
for idx in range (36):
    folder_path = f'dataset/logs/t_{idx:02d}'
    traj_rgb_path = os.path.join(folder_path, 'contact')
    traj_rgb_lst = folder2filelist(traj_rgb_path)
    traj_rgb_lst.sort()

    mode  = "manaul"
    if mode == "manaul":
        mask = get_traj_mask(traj_rgb_lst)
        # init    
        root = tk.Tk()

        # load image
        image = Image.open(mask)
        photo = ImageTk.PhotoImage(image)

        # label with image
        l = tk.Label(root, image=photo)
        l.pack()

        l.bind('<Motion>', motion)
        l.bind('<Button-1>', left_click)


        # bind click event to image
        l.bind('<Button-1>', on_click)

        # button with image binded to the same function 
        b = tk.Button(root, image=photo, command=on_click)
        b.pack()

        # button with text closing window
        b = tk.Button(root, text="Close", command=root.destroy)
        b.pack()
            
        # "start the engine"
        root.mainloop()




    if mode == "automatic":

        w = (len(traj_rgb_lst) -1)// (N)
        # print(w , len(traj_rgb_lst), [[w * i] for i in range(N)])
        keyframe_path = [ traj_rgb_lst[w * i] for i in range(N-1)]
        keyframe_path.append(traj_rgb_lst[-1])

        for path in keyframe_path:
            rgb = cv2.imread(path)
            path_ = path.replace('logs', 'keyframes')

            ## make a directory when not exist
            dir_ = os.path.dirname(path_)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            cv2.imwrite(path_, rgb)

