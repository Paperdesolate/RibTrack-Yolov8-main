Rib Detection and Visualization from DICOM Data

This project provides a comprehensive pipeline for processing DICOM medical images, detecting ribs using YOLOv8 object detection, and generating visualization images of individual ribs through 3D reslicing techniques.

Key Features
- DICOM series preprocessing and windowing
- Video generation from DICOM stacks
- Rib detection and tracking with YOLOv8
- 3D visualization of individual ribs using VTK
- Automated rib labeling and numbering

Requirements
- Python 3.8+
- PyDICOM
- SimpleITK
- OpenCV
- VTK
- Ultralytics (YOLOv8)
- NumPy
- MoviePy

Installation
```bash
pip install pydicom opencv-python simpleitk vtk numpy ultralytics moviepy
```

Optimal Model Location
The best performing YOLOv8 model should be placed at:
```
D:/yolov8-main/Track/weights/best.pt
```

 Usage Instructions

1. Configure Path Variables
Modify these paths in the script before running:
```python
dicom_number = 'your_number'            # Patient/study identifier
folder_path = 'your_path'               # Input DICOM directory
save_path = 'your_path_2'               # Output directory
folder_path_save = 'your_path_3'        # Final image save directory
```

2. Run the Main Script
Execute the entire script to process the DICOM data through the complete pipeline:
```python
python rib_processing.py
```

Processing Pipeline
1. **DICOM Preparation**
   - Renames files to ensure .dcm extension
   - Filters DICOM series by SeriesNumber
   - Sorts slices by InstanceNumber
   - Applies Hounsfield Unit conversion and windowing

2. **Video Generation**
   - Creates a video from the DICOM stack
   - Splits video into left and right halves

3. **Rib Detection & Tracking**
   - Uses YOLOv8 with ByteTrack for rib detection
   - Processes left and right sides separately
   - Tracks ribs across frames
   - Filters low-confidence detections

4. **3D Visualization**
   - Calculates optimal reslicing planes for each rib
   - Generates individual rib images using VTK
   - Saves PNG images in anatomical order (R1-R12, L1-L12)

Output Structure
```
save_path/
├── sorted_dicoms/           # Processed DICOM files
├── Rib.mp4                  # Full video
├── Rib_Left.mp4             # Left half video
├── Rib_Right.mp4            # Right half video
├── L*.txt                   # Left rib tracking data
├── R*.txt                   # Right rib tracking data

folder_path_save/
├── L1.png                   # Left rib 1 visualization
├── ...
├── R12.png                  # Right rib 12 visualization
```

Notes
1. Ensure sufficient disk space for intermediate video files
2. The YOLOv8 model should be trained specifically for rib detection
3. Processing time varies with DICOM stack size (expect 10-30 minutes for typical studies)
4. Optimal results require DICOM series with consistent slice spacing

For questions or support, please contact [your contact information].
