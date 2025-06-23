import pydicom  # For reading DICOM files
import cv2  # For image and video processing
import SimpleITK as sitk  # For medical image processing
import numpy as np  # For numerical operations
import random  # For random sampling
from ultralytics import YOLOv8  # For object detection and tracking
import shutil  # For file operations
import vtkmodules.all as vtk  # For 3D visualization and image processing
import os  # For operating system interactions
import math  # For mathematical operations
from moviepy.editor import VideoFileClip  # For video editing
import re  # For regular expressions
import time  # For time-related operations

# Define paths and initialize variables
dicom_number = 'your_number'  # Identifier for the DICOM series
folder_path = 'your_path'  # Path to input DICOM files
save_path = 'your_path_2'  # Path for output files
save_sorted_dir = os.path.join(save_path, 'sorted_dicoms')  # Directory for sorted DICOMs

# Clear output directory to ensure clean execution
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Delete file or link
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Delete directory
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


def rename_dicom_files(folder_path):
    """Rename DICOM files to ensure they have .dcm extension"""
    renamed_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Add .dcm extension if missing
            if not file.lower().endswith(".dcm"):
                new_file_path = file_path + ".dcm"
                os.rename(file_path, new_file_path)
                renamed_files.append(new_file_path)
            else:
                renamed_files.append(file_path)
    return renamed_files


def get_all_dicom_files(folder_path):
    """Recursively collect all DICOM file paths in directory"""
    dicom_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            dicom_files.append(os.path.join(root, file))
    return dicom_files


def translate(folder_path):
    """Process DICOM series: filter, sort, decompress, and apply windowing"""
    reader = sitk.ImageSeriesReader()
    image_paths = get_all_dicom_files(folder_path)
    filtered_paths = []

    # Filter DICOM files by series number
    for path in image_paths:
        ds = pydicom.dcmread(path)
        series_num = ds.SeriesNumber
        if series_num == '2':  # Select specific series
            filtered_paths.append(path)

    if not filtered_paths:
        print("No rib scan DICOM data found")
        return None

    # Sort DICOMs by instance number
    filtered_paths = sorted(filtered_paths, key=lambda x: pydicom.dcmread(x).InstanceNumber)

    # Save sorted DICOMs to new directory
    if os.path.exists(save_sorted_dir):
        shutil.rmtree(save_sorted_dir)
    os.makedirs(save_sorted_dir, exist_ok=True)

    for idx, file_path in enumerate(filtered_paths):
        new_filename = f"{idx:04d}.dcm"
        dest_path = os.path.join(save_sorted_dir, new_filename)
        shutil.copy2(file_path, dest_path)

    filtered_paths = [os.path.join(save_sorted_dir, f) for f in os.listdir(save_sorted_dir)
                      if f.endswith(".dcm")]

    # Decompress DICOM files if needed
    for dicom_path in filtered_paths:
        try:
            ds = pydicom.dcmread(dicom_path, force=True)
            if ds.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.70":
                ds.decompress()
                ds.save_as(dicom_path)
        except Exception as e:
            print(f"File processing failed: {dicom_path} - Error: {e}")

    # Read and process DICOM series
    reader.SetFileNames(filtered_paths)
    ds = pydicom.dcmread(filtered_paths[0])
    image = reader.Execute()
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept

    # Convert to Hounsfield Units and apply windowing
    image_array = sitk.GetArrayFromImage(image)
    hu_data = image_array * slope + intercept
    WW2 = 900
    min_value = np.min(image_array)
    WL = -100 if min_value == -2048 else -1000
    WW = WW2 * 2
    High = WL + WW2
    Low = WL - WW2

    # Clip and normalize to 8-bit
    hu_data = np.clip(hu_data, Low, High)
    hu_data = ((((hu_data) - Low) / WW) * 255).astype(np.uint8)
    arr_4d = np.repeat(hu_data[:, :, :, np.newaxis], 3, axis=3)

    return arr_4d, filtered_paths


def numpy_array_to_video(numpy_array, video_out_path):
    """Convert 3D numpy array to MP4 video"""
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]
    out_video_size = (video_width, video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*'avc1'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 30, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)
    video_write_capture.release()


# Process DICOM and generate video
translated_array, dcm_path = translate(folder_path)
video_out_path = save_path + "/" + "Rib.mp4"
numpy_array_to_video(translated_array, video_out_path)


def split_video(video_path):
    """Split video into left and right halves"""
    clip = VideoFileClip(video_path)
    width, height = clip.size
    new_width = width // 2

    # Crop and save left/right videos
    left_clip = clip.crop(x1=0, y1=0, x2=new_width, y2=height)
    left_clip.write_videofile(save_path + "/" + "Rib_Left.mp4")

    right_clip = clip.crop(x1=new_width, y1=0, x2=width, y2=height)
    right_clip.write_videofile(save_path + "/" + "Rib_Right.mp4")


split_video(video_out_path)  # Execute video splitting

# Define paths for YOLO processing
mergefiledir_left = save_path + '/' + 'Rib_Left/labels'
mergefiledir_right = save_path + '/' + 'Rib_Right/labels'
final_file_path_left = save_path + '/Rib_Left/' + 'L.txt'
final_file_path_right = save_path + '/Rib_Right/' + 'R.txt'

# Initialize YOLOv8 model and process videos
Model = YOLOv8("D:/yolov8-main/Track/weights/best.pt")
Model.track(source=save_path + "/" + "Rib_Right.mp4", show=True, save=True,
            save_txt=True, imgsz=512, save_conf=True, tracker="bytetrack.yaml",
            name='D:/yolov8-main/runs/detect/Rib/Rib_Right')
Model.track(source=save_path + "/" + "Rib_Left.mp4", show=True, save=True,
            save_txt=True, imgsz=512, save_conf=True, tracker="bytetrack.yaml",
            name='D:/yolov8-main/runs/detect/Rib/Rib_Left')

# Process tracking results for left ribs
lines_by_number1 = {i: [] for i in range(1, 31)}
with open(final_file_path_left, 'w', encoding='utf8') as final_file1:
    filenames1 = os.listdir(mergefiledir_left)
    for filename in filenames1:
        file_number1 = filename.split('_')[-1].split('.')[0]
        filepath1 = os.path.join(mergefiledir_left, filename)
        with open(filepath1, encoding='utf8') as file1:
            for line in file1:
                parts1 = line.strip().split(" ")
                parts1[0] = '0'  # Set class ID
                del parts1[3:6]  # Remove unnecessary values
                parts1[1] = str(float(parts1[1]) * 256 + 256)  # Adjust X coordinate
                parts1[2] = str(float(parts1[2]) * 512)  # Adjust Y coordinate
                parts1.insert(-1, file_number1)  # Insert frame number
                new_line1 = " ".join(parts1)
                final_file1.write(new_line1 + '\n')
                last_number1 = float(parts1[-1])
                if last_number1 in lines_by_number1:
                    lines_by_number1[last_number1].append((float(file_number1), new_line1))

# Process tracking results for right ribs (similar to left)
lines_by_number2 = {i: [] for i in range(1, 31)}
with open(final_file_path_right, 'w', encoding='utf8') as final_file2:
    filenames2 = os.listdir(mergefiledir_right)
    for filename in filenames2:
        file_number2 = filename.split('_')[-1].split('.')[0]
        filepath2 = os.path.join(mergefiledir_right, filename)
        with open(filepath2, encoding='utf8') as file2:
            for line in file2:
                parts2 = line.strip().split(" ")
                parts2[0] = '1'  # Different class ID for right
                del parts2[3:6]
                parts2[1] = str(float(parts2[1]) * 256)
                parts2[2] = str(float(parts2[2]) * 512)
                parts2.insert(-1, file_number2)
                new_line2 = " ".join(parts2)
                final_file2.write(new_line2 + '\n')
                last_number2 = float(parts2[-1])
                if last_number2 in lines_by_number2:
                    lines_by_number2[last_number2].append((float(file_number2), new_line2))


def write_lines_to_files_sorted_and_filtered(lines_dict, prefix):
    """Write filtered and sampled tracking data to files"""
    for number, lines in lines_dict.items():
        file_path = os.path.join(save_path, f'{prefix}{number}.txt')
        sorted_lines = sorted(lines, key=lambda x: x[0])  # Sort by frame number

        # Skip if insufficient data points
        if len(sorted_lines) <= 12:
            continue

        # Sample representative points
        lines_to_keep_count = len(sorted_lines) - len(sorted_lines) % 5
        sorted_lines = sorted_lines[:lines_to_keep_count]
        parts = [sorted_lines[i:i + lines_to_keep_count // 5] for i in
                 range(0, lines_to_keep_count, lines_to_keep_count // 5)]
        selected_lines = [random.choice(part)[1] for part in parts[1:4]]  # Middle sections

        # Write to file
        with open(file_path, 'w', encoding='utf8') as file:
            for line in selected_lines:
                file.write(line + '\n')


# Process left and right tracking data
write_lines_to_files_sorted_and_filtered(lines_by_number1, 'L')
write_lines_to_files_sorted_and_filtered(lines_by_number2, 'R')


def rename_files_with_prefix(save_path, prefix):
    """Rename files with sequential numbering"""
    files = [f for f in os.listdir(save_path) if re.match(rf'{prefix}\d+\.txt', f)]
    sorted_files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))

    # Temporary rename to avoid conflicts
    temp_prefix = "temp_"
    for file in sorted_files:
        os.rename(os.path.join(save_path, file), os.path.join(save_path, temp_prefix + file))

    # Final rename with sequential numbers
    start_number = 12  # Start from highest rib number
    for file in sorted_files:
        new_name = f"{prefix}{start_number}.txt"
        os.rename(os.path.join(save_path, temp_prefix + file), os.path.join(save_path, new_name))
        start_number -= 1  # Decrement for anatomical order


rename_files_with_prefix(save_path, 'L')
rename_files_with_prefix(save_path, 'R')


def process_rib_files(prefix):
    """Main processing pipeline for rib detection and image generation"""
    directory_path = 'D:/yolov8-main/runs/detect/Rib'
    side = "Left" if prefix == 'L' else "Right"
    labels_path = f'{directory_path}/Rib_{side}/{prefix}.txt'

    # File paths for intermediate results
    output_max_path = f'D:/yolov8-main/{prefix}finish.txt'
    output_min_path = f'D:/yolov8-main/{prefix}start.txt'
    output_path = f'D:/yolov8-main/{prefix}close_points.txt'

    # Read and process tracking data
    values_dict = {}
    for file_name in os.listdir(directory_path):
        if file_name.startswith(prefix) and file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'r') as file:
                first_line = file.readline().strip()
                last_value = first_line.split()[-1]
                if len(values_dict) < 20:  # Limit number of entries
                    values_dict[file_name] = last_value

    # Find extreme points in tracking data
    with open(output_max_path, 'w') as max_file, open(output_min_path, 'w') as min_file:
        for key in values_dict:
            target_value = values_dict[key]
            max_value = -float('inf')
            min_value = float('inf')
            max_line = None
            min_line = None

            with open(labels_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if parts[-1] == target_value:
                        second_last_value = float(parts[-2])
                        if second_last_value > max_value:
                            max_value = second_last_value
                            max_line = line.strip()
                        if second_last_value < min_value:
                            min_value = second_last_value
                            min_line = line.strip()

            if max_line:
                max_file.write(max_line + '\n')
            if min_line:
                min_file.write(min_line + '\n')

    # Calculate Euclidean distance between points
    def calculate_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Find close point pairs
    start_points = []
    with open(output_min_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x = float(parts[1])
            y = float(parts[2])
            start_points.append((x, y, line.strip()))

    finish_points = []
    with open(output_max_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x = float(parts[1])
            y = float(parts[2])
            finish_points.append((x, y, line.strip()))

    with open(output_path, 'w') as output_file:
        for x1, y1, full_line in start_points:
            for x2, y2, _ in finish_points:
                if calculate_distance(x1, y1, x2, y2) < 5:  # Distance threshold
                    last_value = full_line.split()[-1]
                    output_file.write(f"{last_value}\n")
                    break

    # Filter out low-quality detections
    target_numbers = set()
    with open(output_path, 'r') as file:
        for line in file:
            number = line.strip()
            target_numbers.add(number)

    # Delete corresponding files
    for file_name in os.listdir(directory_path):
        if file_name.startswith(prefix) and file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            try:
                with open(file_path, 'r') as file:
                    first_line = file.readline().strip()
                    last_number = first_line.split()[-1]
                if last_number in target_numbers:
                    time.sleep(1)  # Allow file system to release handle
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_name}: {str(e)}")


# Execute processing for left and right ribs
process_rib_files('L')
process_rib_files('R')


def final_rename_files(save_path, prefix):
    """Final renaming with sequential numbering"""
    files = [f for f in os.listdir(save_path) if re.match(rf'{prefix}-?\d+\.txt', f)]
    sorted_files = sorted(files, key=lambda x: int(re.search(r'-?\d+', x).group()))

    # Temporary rename to avoid conflicts
    temp_prefix = "temp_"
    for file in sorted_files:
        os.rename(os.path.join(save_path, file), os.path.join(save_path, temp_prefix + file))

    # Final sequential numbering
    start_number = 1
    for file in sorted_files:
        new_name = f"{prefix}{start_number}.txt"
        os.rename(os.path.join(save_path, temp_prefix + file), os.path.join(save_path, new_name))
        start_number += 1


final_rename_files(save_path, 'L')
final_rename_files(save_path, 'R')

# Generate rib images from DICOM and tracking data
rib_numbers = ['L' + str(i) for i in range(1, 13)] + ['R' + str(i) for i in range(1, 13)]
for Rib_number in rib_numbers:
    # Setup VTK rendering environment
    renX = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renX)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Read DICOM series
    v16 = vtk.vtkDICOMImageReader()
    v16.SetDirectoryName(save_sorted_dir)
    v16.Update()

    # Get DICOM metadata
    reader = sitk.ImageSeriesReader()
    image_path = reader.GetGDCMSeriesFileNames(save_sorted_dir)
    reader.SetFileNames(image_path)
    image = reader.Execute()
    img = image[0]
    min_value = np.min(img)
    spacing = image.GetSpacing()  # Pixel spacing (width, height, depth)
    dcm_number = sum(1 for file in os.listdir(save_sorted_dir) if file.endswith('.dcm'))


    # Apply windowing to DICOM
    def adjust_window_level_width(input_image, level, width):
        windowed_image = vtk.vtkImageMapToWindowLevelColors()
        windowed_image.SetWindow(width)
        windowed_image.SetLevel(level)
        windowed_image.SetInputConnection(input_image.GetOutputPort())
        return windowed_image


    window_width = 1800
    window_level = -100
    adjusted_image = adjust_window_level_width(v16, window_level, window_width)

    # Process tracking points
    try:
        with open('D:/yolov8-main/runs/detect/Rib/' + Rib_number + '.txt', 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Warning: File {Rib_number} not found, skipping")
        continue

    # Extract and transform coordinates
    txt_values = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            txt_values.extend([float(parts[1]), float(parts[2]), float(parts[3])])

    # Process first three points (A, B, C)
    txt_1, txt_2, txt_3, txt_4, txt_5, txt_6, txt_7, txt_8, txt_9 = txt_values[:9]
    A_x = txt_1 * spacing[0]
    A_y = (512 - txt_2) * spacing[1]  # Flip Y-axis
    A_z = (dcm_number - txt_3) * spacing[2]  # Adjust for slice order
    B_x = txt_4 * spacing[0]
    B_y = (512 - txt_5) * spacing[1]
    B_z = (dcm_number - txt_6) * spacing[2]
    C_x = txt_7 * spacing[0]
    C_y = (512 - txt_8) * spacing[1]
    C_z = (dcm_number - txt_9) * spacing[2]
    A = (A_x, A_y, A_z)
    B = (B_x, B_y, B_z)
    C = (C_x, C_y, C_z)


    # Calculate normal vector for reslicing plane
    def calculate_normal_vector_with_conditions(A, B, C):
        AB = np.array([B[0] - A[0], B[1] - A[1], B[2] - A[2])
        AC = np.array([C[0] - A[0], C[1] - A[1], C[2] - A[2])
        normal = np.cross(AB, AC)
        if normal[2] < 0:  # Ensure normal points upward
            normal = -normal
        return normal / np.linalg.norm(normal)  # Normalize


    def rotation_matrix_from_vectors(vec1, vec2):
        """Calculate rotation matrix between two vectors"""
        vec1_u = vec1 / np.linalg.norm(vec1)
        vec2_u = vec2 / np.linalg.norm(vec2)
        cross_product = np.cross(vec1_u, vec2_u)
        dot_product = np.dot(vec1_u, vec2_u)
        cross_product_matrix = np.array([
            [0, -cross_product[2], cross_product[1]],
            [cross_product[2], 0, -cross_product[0]],
            [-cross_product[1], cross_product[0], 0]
        ])
        return np.eye(3) + cross_product_matrix + cross_product_matrix.dot(cross_product_matrix) * (
                    1 / (1 + dot_product))


    # Calculate rotation matrix
    normal_vector = calculate_normal_vector_with_conditions(A, B, C)
    rotation_matrix = rotation_matrix_from_vectors([0, 0, 1], normal_vector)

    # Build transformation matrix
    center = [A_x, A_y, A_z]
    resliceAxes = vtk.vtkMatrix4x4()
    resliceAxes.DeepCopy((
        rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], center[0],
        rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], center[1],
        rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], center[2],
        0, 0, 0, 1
    ))

    # Reslice image along calculated plane
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(adjusted_image.GetOutputPort())
    reslice.SetOutputDimensionality(2)
    reslice.SetAutoCropOutput(True)
    reslice.SetResliceAxes(resliceAxes)
    reslice.SetInterpolationModeToLinear()
    reslice.SetOutputExtent(1, 640, 1, 640, 0, 0)  # Output image size
    reslice.SetBackgroundColor(0.0, 0.0, 0.0, 0.0)  # Transparent background
    reslice.Update()

    # Save resliced image as PNG
    folder_path_save = f"your_path_3/" + str(dicom_number)
    if not os.path.exists(folder_path_save):
        os.makedirs(folder_path_save)
    output_filename = f"{folder_path_save}/{Rib_number}.png"

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(reslice.GetOutput())
    writer.Write()
    print(f"Saved {output_filename}")