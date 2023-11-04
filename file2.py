import os
import sys
import warnings
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QLabel, QSpinBox, QWidget

class ImageCompressor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Compressor')
        self.setGeometry(100, 100, 400, 200)

        self.dir1 = None
        self.dir2 = None
        self.quant = None

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.btn_select_dir1 = QPushButton('Select Source Directory', self)
        self.btn_select_dir1.clicked.connect(self.select_dir1)
        layout.addWidget(self.btn_select_dir1)

        self.btn_select_dir2 = QPushButton('Select Destination Directory', self)
        self.btn_select_dir2.clicked.connect(self.select_dir2)
        layout.addWidget(self.btn_select_dir2)

        self.quant_label = QLabel("Quantization (2-256):")
        layout.addWidget(self.quant_label)
        self.quant_spinbox = QSpinBox(self)
        self.quant_spinbox.setRange(2, 256)
        layout.addWidget(self.quant_spinbox)

        self.compress_button = QPushButton('Compress Images', self)
        self.compress_button.clicked.connect(self.compress_images)
        layout.addWidget(self.compress_button)

        self.central_widget.setLayout(layout)

    def select_dir1(self):
        self.dir1 = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        if self.dir1:
            print(f"Source Directory: {self.dir1}")

    def select_dir2(self):
        self.dir2 = QFileDialog.getExistingDirectory(self, "Select Destination Directory")
        if self.dir2:
            print(f"Destination Directory: {self.dir2}")

    def compress_images(self):
        if not (self.dir1 and self.dir2):
            print("Select both source and destination directories.")
            return

        self.quant = self.quant_spinbox.value()
        if self.quant < 2:
            self.quant = 2

        total_count, total_src_size, total_dst_size = 0, 0, 0

        for fname in os.listdir(self.dir1):
            src_fname = os.path.join(self.dir1, fname)
            dst_fname = os.path.join(self.dir2, fname)

            try:
                src_size = os.path.getsize(src_fname)
                img_array2d = self.loadImageFileAsRGBArray(src_fname)
            except:
                print(f'{fname} skip')
                continue

            h, w, _ = img_array2d.shape

            print(f'Compressing {fname}...')

            n_iter = self.saveImageAsPalettePNGusingKMeans(self.quant, img_array2d, dst_fname)

            self.callOptiPNG(dst_fname)

            dst_size = os.path.getsize(dst_fname)

            if src_size <= dst_size:
                if self.dir1 != self.dir2:
                    os.remove(dst_fname)
                dst_fname = os.path.join(self.dir2, fname)
                self.copy_file(src_fname, dst_fname)
                dst_size = src_size
                print(f'{fname} ({w}x{h})   Just copied')
            else:
                print(f'{fname} ({w}x{h})   KMeans iterations={n_iter}   {src_size}B -> {dst_size}B   {100.0 * (1.0 - dst_size / src_size):.2f}% off!')

            total_count += 1
            total_src_size += src_size
            total_dst_size += dst_size

        if total_count > 0:
            print(f'Total {total_count} files compressed    {total_src_size}B -> {total_dst_size}B   {100.0 * (1.0 - total_dst_size / total_src_size):.2f}% off!')
        else:
            print('No input files')

    def loadImageFileAsRGBArray(self, fname):
        img = Image.open(fname)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            img_rgb = img.convert('RGB')
        img.close()
        return np.asarray(img_rgb)

    def saveImageAsPalettePNGusingKMeans(self, quant, img_array2d, fname):
        h, w, _ = img_array2d.shape
        kmeans_n_clusters = min(quant, w * h)
        img_array1d = img_array2d.reshape((-1, 3))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            kmeans_res = MiniBatchKMeans(n_clusters=kmeans_n_clusters, init='k-means++', max_iter=100,
                                         batch_size=32768, compute_labels=True, random_state=0).fit(img_array1d)

        palette_array = np.array(kmeans_res.cluster_centers_ + 0.5, dtype=np.uint8)
        img_palette_array1d = palette
