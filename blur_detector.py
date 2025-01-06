import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import json
import random
import argparse
import itertools 

class BlurDetector:
    # Class for detecting motion blur in frames, and implement blur-aware downsampling,
    # i.e. maintaining the images with the sharpest content
    def __init__(self, path, kernel = 5, precomputed = False):
        # path: path to images (str)
        # kernel: kernel size (int)
        # precomputed: whether blurriness values have been precomputed (boolean)
        self.path = path
        self.kernel = 5
        assert self.path is not str, "Expected input to be directory path as string"
        
        self.results_path = os.path.join(os.path.dirname(__file__), "results")
        print(f"Saving results to {self.results_path}")
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path)
        
        self.samples_path = os.path.join(os.path.dirname(__file__), "samples")
        print(f"Saving samples to {self.samples_path}")
        if not os.path.isdir(self.samples_path):
            os.makedirs(self.samples_path)
        
        self.blur_values_path = os.path.join(self.results_path, "blur_values.json")
        self.blur_values = None
        
        self.img_list = None
        self.generate_img_list()

        if precomputed:
            self.load_blur_values()

    def load_blur_values(self):
        # Load precomputed blur values from a file
        with open(self.blur_values_path, 'r') as f:
            self.blur_values = json.load(f)
    
    def img_blurriness(self, img):
        # Compute blur value for an image
        # img: BGR image (8-bit uint numpy array)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_32F, self.kernel).var()
    
    def generate_img_list(self): 
        # Generate list of images to process
        self.img_list = sorted(os.listdir(self.path))

    def compute_blur_values(self): 
        # Compute blur values for the provided files
        blur_values = {}
        for file in self.img_list:
            img_path = os.path.join(self.path, file)
            try:
                img = cv2.imread(img_path)
                blur_values[file] = float(self.img_blurriness(img))
            except cv2.error:
                print(f"Skipping {file} since it is not a valid image")

        self.blur_values = blur_values
        with open(self.blur_values_path, 'w') as f:
            json.dump(blur_values, f)

    def plot_blur_values(self):
        # Plot computed blur values
        assert self.blur_values is not None, "Blur values not computed"
        values = list(self.blur_values.values())
        plt.hist(values, bins=20)
        plt.xlabel("Sharpness (variance of Laplacian)")
        plt.ylabel("Number of samples")
        plt.grid()
        plt.show()
    
    def visualise(self, image, blur_value):
        # Visualise an image with the corresponding blur value
        # image: image (8-bit uint numpy array)
        # blur_value: blurriness value (float)
        text = str(round(blur_value, 1))
        position = (int(image.shape[0] / 10), int(image.shape[1] / 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 12
        outline_color = (0, 0, 0)  
        text_color = (255, 255, 255)  
        thickness = 12 
        
        for offset in [-4, 0, 4]:  
            cv2.putText(image, text, (position[0] + offset, position[1]), font, font_scale, outline_color, thickness)
            cv2.putText(image, text, (position[0], position[1] + offset), font, font_scale, outline_color, thickness)
        
        cv2.putText(image, text, position, font, font_scale, text_color, thickness) 
        return image

    def save_img(self, image, sample):
        # Save image samples to the samples-folder
        # image: image (8-bit uint numpy array)
        # sample: name of the sample file (str)
        path = os.path.join(self.samples_path, sample)
        print(f"Saving sample to {path}")
        cv2.imwrite(path, image)

    def visualise_random(self, n = 10):
        # Visualise a provided number of image samples with blur values
        # n: number of images to visualise (int)
        assert self.blur_values is not None, "Blur values not computed"
        samples = random.sample(self.img_list, n) 
        for sample in samples:
            img_path = os.path.join(self.path, sample)
            try:
                image = cv2.imread(img_path)
                blur_value = self.blur_values[sample]
                vis = self.visualise(image, blur_value)
                self.save_img(vis, sample)
            except cv2.error:
                print(f"Skipping {sample} due to a problem in visualisation")

    def visualise_random_sequence(self, n = 10):
        # Visualise a random sequence of images with blur values
        # n: number of images to visualise (int)
        assert self.blur_values is not None, "Blur values not computed"
        images = []
        start_idx = random.randint(0, len(self.img_list) - n) 
        for i in range(start_idx, start_idx + n):
            sample = self.img_list[i]
            img_path = os.path.join(self.path, sample)
            try:
                image = cv2.imread(img_path)
                blur_value = self.blur_values[sample]
                vis = self.visualise(image, blur_value)
                self.save_img(vis, sample)
            except cv2.error:
                print(f"Skipping {sample} due to a problem in visualisation")

    def downsample(self, n = 10):
        # Downsample the sequential data, taking the sample with lowest amount of blur
        # n: sub-sequence length to reduce to a single image (i.e. downsampling to 1/n amount of images)
        assert self.blur_values is not None, "Blur values not computed"
        selected_imgs = []
        img_list_chunks = list(itertools.batched(iter(self.img_list), n))
        for chunk in img_list_chunks:
            chunk_blur_values = []
            for img in chunk:
                chunk_blur_values.append(self.blur_values[img])
            selected_imgs.append(chunk[np.argmax(chunk_blur_values)])
        np.savetxt(os.path.join(self.results_path, "downsampled.txt"), selected_imgs, fmt="%s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an image collage.")
    parser.add_argument("--images", "-img", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--precomputed", "-pre", action="store_true", help="Flag to indicate whether blur values have been precomputed")
    parser.add_argument("--downsample", "-ds", action="store_true", help="Flag to perform downsampling.")

    args = parser.parse_args()
    
    if args.precomputed:
        bd = BlurDetector(args.images, precomputed = True)
    else:
        bd = BlurDetector(args.images)
        bd.compute_blur_values()
    if args.downsample:
        bd.downsample()
    
    #bd.visualise_random_sequence()
    #bd.plot_blur_values()
    #bd.visualise_random()
