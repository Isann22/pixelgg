from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QInputDialog,  # Added for brightness input
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from app.image_processor import ImageProcessor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("./ui/gui.ui", self)
        self.image_processor = ImageProcessor()

        # Setup connections
        self.setup_connections()

    def setup_connections(self):
        self.actionGrayscale.triggered.connect(self.grayscale)
        self.actionOpen.triggered.connect(self.open_image)
        self.btn_clear.clicked.connect(self.clear)

        self.zoom_in.clicked.connect(self.zoom_in_image)
        self.zoom_out.clicked.connect(self.zoom_out_image)
        self.actionBiner.triggered.connect(self.biner_image)
        self.actionSave.triggered.connect(self.save_image)
        self.actionRotasi.triggered.connect(self.rotate_image)
        self.actionBrightness.triggered.connect(self.adjust_brightness)
        self.actionEqual.triggered.connect(self.equalize_histogram)
        self.actionGaussian_Filter.triggered.connect(self.gaussian_filter)
        self.spin_rt.valueChanged.connect(self.rotate_image)
        # self.spin_sc.valueChanged.connect(self.zoom_in_image)
        self.btn_reset.clicked.connect(self.reset_adjustment)
        self.setupBrightnessSlider()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Buka Citra", "", "Image Files (*.jpg *.png *.jpeg *.bmp)"
        )
        if file_path:
            self.image_processor.load_image(file_path)
            self.display_image(self.image_processor.get_image(), label_id=1)

    def save_image(self):
        if self.image_processor.get_image_output() is None:
            QMessageBox.warning(self, "Peringatan", "citra belum di proses")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Citra", "", "JPEG Files (*.jpg);;PNG Files (*.png)"
        )
        if file_path:
            self.image_processor.save_image(file_path)

    def display_image(self, image, label_id=1):
        img = self.image_processor.convert_cv_to_qimage(image)
        if label_id == 1 and hasattr(self, "imglabel"):
            self.imglabel.setPixmap(QPixmap.fromImage(img))
        elif label_id == 2 and hasattr(self, "result"):
            self.result.setPixmap(QPixmap.fromImage(img))

    def grayscale(self):
        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang diproses.")
            return

        if self.image_processor.get_shape_image() == 2:
            QMessageBox.warning(
                self, "Peringatan", "Gambar sudah dalam bentuk grayscale!"
            )
            return

        image = self.image_processor.convert_to_grayscale(
            self.image_processor.get_image()
        )
        self.display_image(image, label_id=2)

    def biner_image(self):
        if self.image_processor.get_image_output() is None:
            QMessageBox.warning(
                self, "Peringatan", "Gambar harus diubah ke grayscale terlebih dahulu!"
            )
            return

        treshold, ok = QInputDialog.getInt(
            self,
            "Biner Gambar",
            "Masukkan jumlah treshold (0-255):",
        )

        if treshold and ok:
            try:
                image = self.image_processor.convert_to_biner(
                    self.image_processor.get_image_output(), treshold
                )
                self.display_image(image, label_id=2)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Gagal membuat gambar biner: {str(e)}"
                )

    def setupBrightnessSlider(self):
        self.slider_br.setValue(0)
        self.lbl_br.setText(str(self.slider_br.value()))
        self.slider_br.valueChanged.connect(self.adjust_brightness)

    def adjust_brightness(self):
        if self.image_processor.get_image_output() is None:
            QMessageBox.warning(
                self, "Peringatan", "Gambar harus diubah ke grayscale terlebih dahulu!"
            )
            return

        try:
            img = (
                self.image_processor.get_image_output()
                if self.image_processor.get_image_output() is not None
                else self.image_processor.get_image()
            )
            brightness = int(self.slider_br.value())
            img = self.image_processor.adjust_brightness(img, brightness)

            self.lbl_br.setText(str(self.slider_br.value()))
            self.display_image(img, 2)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Gagal menyesuaikan brightness: {str(e)}"
            )

    def rotate_image(self):
        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat")
            return

        try:
            self.image_processor.rotate_image(self.spin_rt.value())
            self.display_image(
                self.image_processor.get_image_output(),
                2,
            )
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))

    def equalize_histogram(self):
        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat")
            return

        try:
            img = (
                self.image_processor.get_image_output()
                if self.image_processor.get_image_output() is not None
                else self.image_processor.get_image()
            )
            img = self.image_processor.equalize_histogram(img)
            self.display_image(img, label_id=2)
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Gagal menyesuaikan histogram: {str(e)}"
            )

    def zoom_in_image(self):
        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat")
            return

        try:
            self.image_processor.scale_image(2)
            self.display_image(self.image_processor.get_image_output(), label_id=2)
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))

    def zoom_out_image(self):
        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat")
            return

        try:
            self.image_processor.scale_image(0.5)
            self.display_image(self.image_processor.get_image_output(), label_id=2)
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))

    def gaussian_filter(self):
        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat")
            return

        try:
            self.image_processor.apply_gaussian_filter()
            self.display_image(self.image_processor.get_image_output(), label_id=2)
        except ValueError as e:
            QMessageBox.critical(
                self, "Error", f"Gagal menyesuaikan gaussian filter: {str(e)}"
            )

    def reset_adjustment(self):

        if self.image_processor.get_image() is None:
            QMessageBox.warning(self, "Peringatan", "Tidak ada citra yang dimuat")
            return

        try:
            self.slider_br.setValue(0)
            self.lbl_br.setText(str(self.slider_br.value()))
            self.spin_rt.setValue(0)
            self.image_processor.rotate_image(0)
            self.image_processor.adjust_brightness(
                self.image_processor.get_image_output(), 0
            )
            self.display_image(self.image_processor.get_image(), label_id=2)
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"{str(e)}")

    def clear(self):
        self.result.clear()
        self.image_processor.clear_image()
