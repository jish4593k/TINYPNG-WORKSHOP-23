import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from skimage.metrics import structural_similarity as SSIM
import numpy as np
import matplotlib.pyplot as plt

def calculate_ssim(img1, img2):
    ssim_r = SSIM(img1[:,:,0], img2[:,:,0], data_range=255.0)
    ssim_g = SSIM(img1[:,:,1], img2[:,:,1], data_range=255.0)
    ssim_b = SSIM(img1[:,:,2], img2[:,:,2], data_range=255.0)
    return (ssim_r + ssim_g + ssim_b) / 3.0

def load_image(file_path):
    try:
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except:
        return None

def browse_dir():
    dir_path = filedialog.askdirectory()
    if dir_path:
        dir_var.set(dir_path)
        compare_images()

def compare_images():
    dir1 = dir_var1.get()
    dir2 = dir_var2.get()
    
    if not (dir1 and dir2):
        result_label.config(text="Select both directories.")
        return
    
    image_list1 = [f for f in os.listdir(dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    image_list2 = [f for f in os.listdir(dir2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]

    result_label.config(text="")
    canvas.delete("all")
    
    for fname1 in image_list1:
        fname2 = fname1
        if fname1 not in image_list2:
            result_label.config(text=f"{fname1} has no corresponding file in {dir2}. Skipping.")
            continue

        img1 = load_image(os.path.join(dir1, fname1))
        img2 = load_image(os.path.join(dir2, fname2))

        if img1 is None or img2 is None:
            result_label.config(text=f"Failed to load {fname1} or {fname2}. Skipping.")
            continue
        
        ssim_score = calculate_ssim(img1, img2)
        result_label.config(text=f"SSIM Score for {fname1} and {fname2}: {ssim_score:.5f}")

        img1_resized = cv2.resize(img1, (200, 200))
        img2_resized = cv2.resize(img2, (200, 200))

        img_concatenated = cv2.hconcat([img1_resized, img2_resized])
        img_concatenated_rgb = cv2.cvtColor(img_concatenated, cv2.COLOR_BGR2RGB)
        img_concatenated_tk = Image.fromarray(img_concatenated_rgb)

        img_canvas = ImageTk.PhotoImage(img_concatenated_tk)
        canvas.create_image(10, 10, anchor="nw", image=img_canvas)
        canvas.image = img_canvas

root = tk.Tk()
root.title("Image Comparison Tool")

frame1 = ttk.Frame(root)
frame1.pack(padx=10, pady=10, fill="both", expand=True)

label1 = ttk.Label(frame1, text="Directory 1:")
label1.grid(row=0, column=0, padx=(0, 5))
dir_var1 = tk.StringVar()
entry1 = ttk.Entry(frame1, textvariable=dir_var1)
entry1.grid(row=0, column=1, padx=(0, 5))

browse_button1 = ttk.Button(frame1, text="Browse", command=browse_dir)
browse_button1.grid(row=0, column=2)

label2 = ttk.Label(frame1, text="Directory 2:")
label2.grid(row=1, column=0, padx=(0, 5))
dir_var2 = tk.StringVar()
entry2 = ttk.Entry(frame1, textvariable=dir_var2)
entry2.grid(row=1, column=1, padx=(0, 5))

browse_button2 = ttk.Button(frame1, text="Browse", command=browse_dir)
browse_button2.grid(row=1, column=2)

compare_button = ttk.Button(frame1, text="Compare", command=compare_images)
compare_button.grid(row=2, column=0, columnspan=3, pady=10)

frame2 = ttk.Frame(root)
frame2.pack(fill="both", expand=True)

result_label = ttk.Label(frame2, text="")
result_label.pack(pady=10)

canvas = tk.Canvas(frame2, width=420, height=200)
canvas.pack()

root.mainloop()
