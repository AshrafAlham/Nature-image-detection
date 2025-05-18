import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QDialog,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QLabel,
    QMessageBox,
    QCheckBox,
)
from PyQt5.QtGui import QPixmap, QImage

from PyQt5 import uic
from tensorflow.keras.models import load_model
from functools import partial
import warnings

warnings.filterwarnings("ignore")


class ImageClassifier(QDialog):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        uic.loadUi(
            r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\Image_Classification.ui",
            self,
        )
        # resizing window of GUI
        self.resize(1000, 800)

        # Buttons
        self.upload_button = self.findChild(QPushButton, "Browse")  # "Upload Image" button
        self.upload_button.clicked.connect(self.load_image)

        self.camera_button = self.findChild(QPushButton, "Camera")  # "Camera" button
        self.camera_button.clicked.connect(self.capture_from_camera)

        self.shape_button = self.findChild(QPushButton, "Image_Shape")  # "Image Shape" button
        self.shape_button.clicked.connect(self.show_image_shape)

        self.edge_detection_button = self.findChild(QPushButton, "Edge_Detection")
        self.edge_detection_button.clicked.connect(self.detect_edges)

        # Predict buttons for individual models
        self.predict_vgg16_button = self.findChild(QPushButton, "VGG16")
        self.predict_vgg16_button.clicked.connect(partial(self.predict_with_model, "vgg16"))

        self.predict_vgg19_button = self.findChild(QPushButton, "VGG19")
        self.predict_vgg19_button.clicked.connect(partial(self.predict_with_model, "vgg19"))

        self.predict_custom_button = self.findChild(QPushButton, "Custom")
        self.predict_custom_button.clicked.connect(partial(self.predict_with_model, "custom"))

        self.predict_inception_button = self.findChild(QPushButton, "Inception")
        self.predict_inception_button.clicked.connect(partial(self.predict_with_model, "inception"))

        self.predict_ResNet_button = self.findChild(QPushButton, "ResNet")
        self.predict_ResNet_button.clicked.connect(partial(self.predict_with_model, "ResNet"))

        # Predict button for ensemble
        self.ensemble_button = self.findChild(QPushButton, "Best_Model")
        self.ensemble_button.clicked.connect(self.predict_with_ensemble)

        # Checkboxes for model selection in ensemble
        self.vgg16_checkbox = self.findChild(QCheckBox, "checkBox_vgg16")
        self.vgg19_checkbox = self.findChild(QCheckBox, "checkBox_vgg19")
        self.custom_checkbox = self.findChild(QCheckBox, "checkBox_custom")
        self.inception_checkbox = self.findChild(QCheckBox, "checkBox_inception")
        self.ResNet_checkbox = self.findChild(QCheckBox, "checkBox_ResNet")

        # Image viewer
        self.image_viewer = self.findChild(QGraphicsView, "graphicsView")
        self.scene = QGraphicsScene(self)
        self.image_viewer.setScene(self.scene)

        # Result labels
        self.result_label = self.findChild(QLabel, "label")
        self.result_label_2 = self.findChild(QLabel, "label_2")

        # Class names mapping
        self.classes = {0: "buildings", 1: "forest", 2: "glacier", 3: "mountain", 4: "sea", 5: "street"}

        # Initialize image path
        self.image_path = None

        # Initialize models as None (lazy loading)
        self.vgg16_model = None
        self.vgg19_model = None
        self.custom_model = None
        self.inception_model = None
        self.ResNet_model = None

        # Enable drag and drop
        self.setAcceptDrops(True)

    def load_image(self):
        """Load an image from the file system."""
        self.result_label.setText("Loading image....")
        QApplication.processEvents()

        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files ()", options=options
        )
        if self.image_path:
            self.display_image(self.image_path)

    def detect_edges(self):
        """Perform edge detection on the loaded image and resize the result to fit the QGraphicsView."""
        if not self.image_path:
            self.result_label.setText("Please load an image first.")
            return

        try:
            # Load the original image
            original_img = cv2.imread(self.image_path)
            if original_img is None:
                raise ValueError("Unable to load image.")

            # Convert the original image to grayscale for edge detection
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)  # Adjust thresholds as needed

            # Get the size of the QGraphicsView
            view_width = self.image_viewer.width()
            view_height = self.image_viewer.height()

            # Resize the edge-detected image to fit the QGraphicsView while maintaining aspect ratio
            h, w = edges.shape
            scale = min(view_width / w, view_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            edges_resized = cv2.resize(edges, (new_width, new_height))

            # Convert the resized edges image to a QImage
            height, width = edges_resized.shape
            bytes_per_line = width
            q_img = QImage(edges_resized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            # Display the resized edges image in the QGraphicsView
            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(q_img))
            self.image_viewer.setScene(self.scene)
            self.result_label.setText("Edge detection applied successfully.")
        except Exception as e:
            self.result_label.setText(f"Error during edge detection: {e}")

    def capture_from_camera(self):
        """Capture an image from the camera."""
        self.result_label.setText("Loading camera...")
        QApplication.processEvents()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Unable to open camera.")
            return

        ret, frame = cap.read()
        if ret:
            self.image_path = "captured_image.jpg"
            cv2.imwrite(self.image_path, frame)
            self.display_image(self.image_path)
        else:
            QMessageBox.warning(self, "Camera Error", "Failed to capture image.")

        cap.release()

    def display_image(self, image_path):
        """Display the image in the QGraphicsView, resizing it to fit the view while maintaining aspect ratio."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Unable to load image.")

            # Get the size of the QGraphicsView
            view_width = self.image_viewer.width()
            view_height = self.image_viewer.height()

            # Resize the image to fit the QGraphicsView while maintaining aspect ratio
            h, w, _ = img.shape
            scale = min(view_width / w, view_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            img_resized = cv2.resize(img, (new_width, new_height))

            # Convert the resized image to a QImage
            height, width, _ = img_resized.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Display the resized image in the QGraphicsView
            self.scene.clear()
            self.scene.addPixmap(QPixmap.fromImage(q_img))
            self.image_viewer.setScene(self.scene)
            self.result_label.setText("Image loaded successfully.")
        except Exception as e:
            self.result_label.setText(f"Error loading image: {e}")

    def preprocess_image(self, image_path):
        """Preprocess the image for model prediction."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Unable to load image.")

            img = cv2.resize(img, (150, 150))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            self.result_label.setText(f"Error processing image: {e}")
            return None

    def load_model_if_needed(self, model_name):
        """Load a model if it hasn't been loaded yet."""
        self.result_label.setText(f"Loading {model_name} model...")
        QApplication.processEvents()

        if model_name == "vgg16" and self.vgg16_model is None:
            self.vgg16_model = load_model(
                r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\models\VGG16.h5"
            )
        elif model_name == "vgg19" and self.vgg19_model is None:
            self.vgg19_model = load_model(
                r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\models\VGG19.h5"
            )
        elif model_name == "custom" and self.custom_model is None:
            self.custom_model = load_model(
                r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\models\customized_model.h5"
            )
        elif model_name == "inception" and self.inception_model is None:
            self.inception_model = load_model(
                r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\models\InceptionV3_model.h5"
            )
        elif model_name == "ResNet" and self.ResNet_model is None:
            self.ResNet_model = load_model(
                r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\models\ResNet152V2_model.h5"
            )

    def predict_with_model(self, model_name):
        """Predict using a single model."""
        if not self.image_path:
            self.result_label.setText("Please load an image first.")
            return

        self.result_label.setText(f"Predicting with {model_name}...")
        QApplication.processEvents()

        self.load_model_if_needed(model_name)
        model = getattr(self, f"{model_name}_model")
        if model is None:
            self.result_label.setText(f"Error: {model_name} model not loaded.")
            return

        img = self.preprocess_image(self.image_path)
        if img is None:
            return

        try:
            predictions = model.predict(img)
            self.display_prediction_result(predictions, model_name)
        except Exception as e:
            self.result_label.setText(f"Prediction error: {e}")

    def predict_with_ensemble(self):
        """Predict using an ensemble of selected models with weighted averaging."""
        if not self.image_path:
            self.result_label.setText("Please load an image first.")
            return

        self.result_label.setText("Predicting with ensemble...")
        QApplication.processEvents()

        img = self.preprocess_image(self.image_path)
        if img is None:
            return

        try:
            selected_models = []

            if self.vgg16_checkbox.isChecked():
                self.load_model_if_needed("vgg16")
                if self.vgg16_model:
                    selected_models.append(("vgg16", self.vgg16_model, 0.3))

            if self.vgg19_checkbox.isChecked():
                self.load_model_if_needed("vgg19")
                if self.vgg19_model:
                    selected_models.append(("vgg19", self.vgg19_model, 0.3))

            if self.custom_checkbox.isChecked():
                self.load_model_if_needed("custom")
                if self.custom_model:
                    selected_models.append(("custom", self.custom_model, 0.2))

            if self.inception_checkbox.isChecked():
                self.load_model_if_needed("inception")
                if self.inception_model:
                    selected_models.append(("inception", self.inception_model, 0.2))

            if self.ResNet_checkbox.isChecked():
                self.load_model_if_needed("ResNet")
                if self.ResNet_model:
                    selected_models.append(("ResNet", self.ResNet_model, 0.2))

            if not selected_models:
                self.result_label.setText("Please select at least one valid model for ensemble.")
                return

            predictions = []
            weights = []

            for model_name, model, weight in selected_models:
                if model:  #
                    pred = model.predict(img)

                    if pred.shape[0] != 1:
                        self.result_label.setText(f"Error: Model {model_name} returned unexpected shape {pred.shape}")
                        return

                    predictions.append(pred[0])  #
                    weights.append(weight)

            if not predictions or not weights:
                self.result_label.setText("Error: No valid predictions obtained.")
                return

            ensemble_predictions = np.average(predictions, axis=0, weights=weights)
            self.display_prediction_result(ensemble_predictions, "ensemble")

        except Exception as e:
            self.result_label.setText(f"Prediction error: {e}")

    def display_prediction_result(self, predictions, model_name):
        """Display the prediction result."""
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        self.result_label.setText(f"Model: {model_name} | Class: {self.classes[class_idx]}")
        self.result_label_2.setText(f"Confidence: {confidence:.2f}%")

    def show_image_shape(self):
        """Display the shape of the loaded image."""
        if self.image_path:
            try:
                img = cv2.imread(self.image_path)
                if img is None:
                    raise ValueError("Unable to load image.")
                shape = img.shape
                self.result_label.setText(f"Image Shape: {shape}")
            except Exception as e:
                self.result_label.setText(f"Error retrieving image shape: {e}")
        else:
            self.result_label.setText("Please load an image first.")


class WelcomeWindow(QDialog):
    def __init__(self):
        super(WelcomeWindow, self).__init__()
        uic.loadUi(
                        r"E:\scientific\Programming\Machine Learning Diploma\Computer Vision\George\CV_final_project\Final_CV_Project\welcome_window.ui", self)

        # Connect the "Start" button (if it exists) to close the welcome window
        start_button = self.findChild(QPushButton, "startButton")
        if start_button:
            start_button.clicked.connect(self.close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Show the welcome window
    welcome_window = WelcomeWindow()
    welcome_window.exec_()
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())
