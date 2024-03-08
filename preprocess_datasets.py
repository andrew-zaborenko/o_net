import dlib
import os
import shutil
import cv2
import numpy as np
import argparse
from tqdm import tqdm

class LandmarksDataset_original:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.landmarks_files = [f for f in os.listdir(root_dir) if f.endswith('.pts')]
        self.detector = dlib.get_frontal_face_detector()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_name)

        landmarks_name = os.path.join(self.root_dir, self.landmarks_files[idx])
        landmarks = self._load_landmarks(landmarks_name)

        return image, landmarks

    def _load_landmarks(self, file_path):
        with open(file_path, 'r') as f:
            landmarks = []
            for line in f.readlines()[3:-1]:  # skip the first 3 and last lines
                x, y = map(float, line.split())
                landmarks.append([x, y])
            return np.array(landmarks)

def merge_datasets(folder1, folder2, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over images and landmarks in the first folder
    for file1 in os.listdir(folder1):
        if file1.endswith('.jpg'):
            basename = os.path.splitext(file1)[0]
            pts_file = os.path.join(folder1, basename + '.pts')
            if os.path.exists(pts_file) and count_landmarks(pts_file) == 68:
                shutil.copy(os.path.join(folder1, file1), os.path.join(output_folder, file1))
                shutil.copy(pts_file, os.path.join(output_folder, basename + '.pts'))

    # Iterate over images and landmarks in the second folder
    for file2 in os.listdir(folder2):
        if file2.endswith('.jpg'):
            basename = os.path.splitext(file2)[0]
            pts_file = os.path.join(folder2, basename + '.pts')
            if os.path.exists(pts_file) and count_landmarks(pts_file) == 68:
                shutil.copy(os.path.join(folder2, file2), os.path.join(output_folder, file2))
                shutil.copy(pts_file, os.path.join(output_folder, basename + '.pts'))

def count_landmarks(pts_file):
    with open(pts_file, 'r') as f:
        num_landmarks = sum(1 for line in f.readlines()[3:-1])
    return num_landmarks
def detect_face_and_resize(image, landmarks):
    detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        # If no face is detected, return None for both face and landmarks
        return None, None

    # Initialize variables to store the best face and the maximum number of landmarks inside a face
    best_face = None
    max_landmarks_inside = 0

    # Iterate over all detected faces to find the one with the most landmarks inside
    for face in faces:
        landmarks_inside = 0
        for x, y in landmarks:
            if face.left() <= x <= face.right() and face.top() <= y <= face.bottom():
                landmarks_inside += 1

        # Update the best face if the current face contains more landmarks
        if landmarks_inside > max_landmarks_inside:
            best_face = face
            max_landmarks_inside = landmarks_inside

    # If no landmarks are inside any detected face, return None for both face and landmarks
    if max_landmarks_inside == 0:
        return None, None

    landmarks_new = []

    for x, y in landmarks:
        if best_face.left() <= x <= best_face.right() and best_face.top() <= y <= best_face.bottom():
            x_new = (x - best_face.left()) / (best_face.right() - best_face.left())
            y_new = (y - best_face.top()) / (best_face.bottom() - best_face.top())
            landmarks_new.append([x_new, y_new])
        else:
            landmarks_new.append([-1, -1])

    # Crop and resize the best face
    try:
        face_image = image[best_face.top():best_face.bottom(), best_face.left():best_face.right()]
        face_image_resized = cv2.resize(face_image, (48, 48))
    except Exception as e:
        # print(f"Error cropping/resizing face: {e}")
        return None, None

    return face_image_resized, np.array(landmarks_new)

def visualize_face_with_landmarks(image, landmarks):
    # Plot the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot the landmarks
    for landmark in landmarks:
        x, y = landmark
        if x != -1 and y != -1:
            plt.scatter(x * image.shape[1], y * image.shape[0], color='red', s=5)
    
    plt.axis('off')
    plt.show()

def visualize_face_with_original_landmarks(image, landmarks_original):
    # Plot the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot the landmarks
    for landmark in landmarks_original:
        x, y = landmark
        if x != -1 and y != -1:
            plt.scatter(x, y, color='red', s=5)
    
    plt.axis('off')
    plt.show()

def extract_landmarks(image, landmarks_normalized):
    # Load face detector and predictor
    detector = dlib.get_frontal_face_detector()
    
    # Convert normalized landmarks to original image coordinates
    def denormalize_landmarks(landmarks, face):
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()
        denormalized_landmarks = []
        for x_norm, y_norm in landmarks:
            if x_norm == -1 and y_norm == -1:
                denormalized_landmarks.append([-1, -1])
            else:
                x_denorm = int((x_norm * face_width) + face.left())
                y_denorm = int((y_norm * face_height) + face.top())
                denormalized_landmarks.append([x_denorm, y_denorm])
        return denormalized_landmarks
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    # Assume only one face is detected
    face = faces[0]
    
    # Extract the face region
    face_image = image[face.top():face.bottom(), face.left():face.right()]
    
    # Resize the face image to (48, 48)
    face_image_resized = cv2.resize(face_image, (48, 48))
    
    # Convert landmarks from normalized to original coordinates
    landmarks_original = denormalize_landmarks(landmarks_normalized, face)
    
    return landmarks_original

def create_dataset(dataset, output_folder, landmarks_file):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the landmarks file for writing
    with open(landmarks_file, 'w') as f:
        for i, sample in tqdm(enumerate(dataset)):
            face, landmarks = detect_face_and_resize(sample[0], sample[1])

            # If face is not detected, skip this sample
            if face is None:
                continue

            # Save cropped face to output folder
            face_filename = os.path.join(output_folder, f"face_{i}.jpg")
            cv2.imwrite(face_filename, face)

            # Write landmarks to landmarks file
            f.write(f"face_{i}.jpg")  # Write image filename
            for landmark in landmarks:
                x, y = landmark
                f.write(f" {x} {y}")
            f.write("\n")


def main(args):
    menpo_train_path = args.menpo_train
    w_train_path = args.w_train
    menpo_test_path = args.menpo_test
    w_test_path = args.w_test

    merge_datasets(menpo_train_path, w_train_path, output_folder='merged_landmarks_train')
    merge_datasets(menpo_test_path, w_test_path, output_folder='merged_landmarks_test')

    dataset_train = LandmarksDataset_original('./merged_landmarks_train/')
    output_folder_train = 'cropped_faces'
    landmarks_file_train = 'landmarks.txt'
    create_dataset(dataset_train, output_folder_train, landmarks_file_train)

    dataset_test = LandmarksDataset_original('./merged_landmarks_test/')
    output_folder_test = 'cropped_faces_test'
    landmarks_file_test = 'landmarks_test.txt'
    create_dataset(dataset_test, output_folder_test, landmarks_file_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--menpo_train", type=str, help="Path to Menpo train dataset")
    parser.add_argument("--w_train", type=str, help="Path to 300W train dataset")
    parser.add_argument("--menpo_test", type=str, help="Path to Menpo test dataset")
    parser.add_argument("--w_test", type=str, help="Path to 300W test dataset")
    args = parser.parse_args()
    main(args)

# python preprocess_datasets.py --menpo_train_path './landmarks_task/Menpo/train/' --w_train_path './landmarks_task/300W/train/' --menpo_test_path './landmarks_task/Menpo/test/' --w_test_path './landmarks_task/300W/test/'
