import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage


class ImageProcessor:
    def __init__(self):
        self.image = None
        self.output = None

    def load_image(self, file_path):
        self.image = cv2.imread(file_path)

    def save_image(self, file_path):
        # Menyimpan citra ke file
        if self.output is not None:
            cv2.imwrite(file_path, self.output)

    def get_image(self):
        return self.image

    def get_image_output(self):
        return self.output

    def get_shape_image(self):
        return len(self.image.shape)

    def get_shape_image_out(self):
        return len(self.output.shape)

    def convert_cv_to_qimage(self, image):
        if image is None:
            return

        if len(image.shape) == 2:
            qformat = QImage.Format_Grayscale8
            img = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                qformat,
            )
        else:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
            img = QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                qformat,
            )
        return img.rgbSwapped()

    def convert_to_grayscale(self, image):
        # Log pixel awal
        self.logging("Pixel awal (sebelum konversi ke grayscale):")
        self.log_pixel_values(self.image)

        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.image[i, j, 0]
                    + 0.587 * self.image[i, j, 1]
                    + 0.114 * self.image[i, j, 2],
                    0,
                    255,
                )

        self.output = gray
        # self.output = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Log pixel setelah perubahan
        self.logging("Pixel setelah konversi ke grayscale:")
        self.log_pixel_values(self.output)

        return self.output

    def convert_to_biner(self, image=None, threshold=127, max_val=255):
        if image is None:
            if self.output is None:
                raise ValueError("Tidak ada gambar yang dimuat")
            image = self.output.copy()

        self.logging("Pixel awal (sebelum konversi ke biner):")
        self.log_pixel_values(image)

        biner_image = np.where(self.output > threshold, max_val, 0).astype(np.uint8)

        self.output = biner_image

        self.logging("Pixel setelah konversi ke biner:")
        self.log_pixel_values(self.output)

        return self.output

    def adjust_brightness(self, image, value):
        self.logging("Pixel awal (sebelum penyesuaian kecerahan):")
        self.log_pixel_values(image)

        if len(image.shape) == 2:
            adjusted_image = cv2.add(image, value)
        else:
            adjusted_image = cv2.add(image, np.array([value]))

        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

        self.logging(f"Pixel setelah penyesuaian kecerahan {value}:")
        self.log_pixel_values(adjusted_image)
        return adjusted_image

    def equalize_histogram(self, image=None):

        self.logging("Histogram sebelum equalization:")
        self.log_pixel_values(image)

        plt.figure(figsize=(10, 5))
        hist_original, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        plt.plot(hist_original, color="blue")
        plt.title("Histogram Original")
        plt.xlabel("Intensitas Pixel")
        plt.ylabel("Frekuensi")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim([0, 256])
        plt.tight_layout()
        plt.show()

        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

        cdf = hist.cumsum()

        cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype("uint8")

        img_equalized = cdf_normalized[image.flatten()]

        if len(image.shape) == 3:  # Gambar warna
            self.output = img_equalized.reshape(image.shape)
        else:  # Gambar grayscale
            self.output = img_equalized.reshape(image.shape)

        # 3. Tampilkan histogram hasil equalisasi
        plt.figure(figsize=(10, 5))
        hist_equalized, _ = np.histogram(
            self.output.flatten(), bins=256, range=[0, 256]
        )
        plt.plot(hist_equalized, color="red")
        plt.title("Histogram Hasil Equalization")
        plt.xlabel("Intensitas Pixel")
        plt.ylabel("Frekuensi")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim([0, 256])
        plt.tight_layout()
        plt.show()

        self.logging("Histogram setelah equalization:")
        self.log_pixel_values(self.output)

        return self.output

    def rotate_image(self, angle):
        if self.image is None:
            raise ValueError("Tidak ada citra yang dimuat")

        image = self.output if self.output is not None else self.image.copy()
        self.logging("Pixel awal (sebelum rotasi):")
        self.log_pixel_values(self.image)

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        self.output = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

        self.logging(f"Rotasi gambar sebesar {angle} derajat")
        self.log_pixel_values(self.output)

        return self.output

    def scale_image(self, scale_factor, interpolation=None):
        if self.output is None:
            if self.image is None:
                raise ValueError("tidak ada citra yang dimuat")
            self.output = self.image.copy()

        height, width = self.output.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        if new_width < 10 or new_height < 10:
            raise ValueError("batas minimum ukuran gambar mencapai batas")

        if new_width > 10000 or new_height > 10000:
            raise ValueError("batas maksimum ukuran gambar mencapai batas")

        if interpolation is None:
            interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC

        self.logging(f"\nSebelum scaling (factor: {scale_factor}):")
        self.log_pixel_values(self.output)

        self.output = cv2.resize(
            self.output, (new_width, new_height), interpolation=interpolation
        )

        self.logging(f"\nSesudah scaling (factor: {scale_factor}):")
        self.log_pixel_values(self.output)

        return self.output

    def apply_gaussian_filter(self, kernel_size=(3, 3), sigma=1.0):
        if self.output is None:
            if self.image is None:
                raise ValueError("No image loaded")
            self.output = self.image.copy()

        self.logging(
            f"\nSebelum Gaussian filter (kernel: {kernel_size}, sigma: {sigma}):"
        )
        self.log_pixel_values(self.output)

        self.output = cv2.GaussianBlur(self.output, kernel_size, sigma)

        self.logging(
            f"\nSesudah Gaussian filter (kernel: {kernel_size}, sigma: {sigma}):"
        )
        self.log_pixel_values(self.output)

        return self.output

    def log_pixel_values(self, image):

        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        pixel_values = image[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3]

        self.logging(f"Nilai pixel pada area 5x5 di tengah gambar:\n{pixel_values}")

    def clear_image(self):
        self.output = None
        self.logging("output dihapus")

    def logging(self, message):
        with open("logging.txt", "a") as logFile:
            logFile.write(f"{message}\n")
        print(f"[LOG] {message}")

    def pixel_cek(self):
        if self.output is None:
            self.log_pixel_values(self.image)
        else:
            self.log_pixel_values(self.output)
