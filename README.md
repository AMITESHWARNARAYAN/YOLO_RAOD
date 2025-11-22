# Road Damage Detection System

YOLOv8-powered system to detect road damage (Cracks, Potholes, Severe Damage) with real-time camera analysis and Google Maps integration.

## Features

- **YOLOv8 Object Detection**: Fast, accurate real-time damage detection
- **Google Maps Integration**: Automated road damage reporting with GPS
- **Real-time Analysis**: Road condition scoring (0-100) with severity classification
- **Production Ready**: FastAPI REST API server for deployment
- **Google Colab Training**: Free GPU access for model training

## Quick Start

### Step 1: Prepare Dataset

```powershell
# Convert classification dataset to YOLO format
python convert_to_yolo_format.py

# Create zip for Colab upload
python prepare_for_colab.py
```

This creates `yolo_data.zip` (~1.7GB) with your dataset.

### Step 2: Train on Google Colab (Recommended)

**Why Colab?** Free GPU (10-100x faster than CPU), no Windows/PyTorch compatibility issues.

1. **Upload to Colab**:
   - Go to https://colab.research.google.com/
   - Upload `train_yolov8_colab.ipynb`
   - Upload `yolo_data.zip` (or save to Google Drive)

2. **Enable GPU**:
   - Runtime → Change runtime type → GPU (T4)

3. **Train**:
   - Run all cells in the notebook
   - Training takes ~30-60 minutes for YOLOv8n
   - Expected accuracy: 88-93% mAP@0.5

4. **Download Model**:
   - Download `trained_model.zip` when complete
   - Extract to your project folder

### Step 3: Test Real-time Detection

```powershell
# Test with webcam
python detect_realtime.py --source 0 --api-key YOUR_GOOGLE_MAPS_KEY

# Test with video file
python detect_realtime.py --source road_video.mp4

# Test with image
python detect_realtime.py --source image.jpg
```

```powershell
# Start REST API server
python api_server.py

# Access API docs at: http://localhost:8000/docs
```

**API Endpoints**:
- `POST /detect` - Upload image for detection
- `POST /detect/url` - Detect from image URL
- `POST /report` - Submit damage report to Google Maps
- `GET /reports` - View recent reports
- `GET /health` - Health check

## Project Structure

```
yolo/
├── convert_to_yolo_format.py   # Convert dataset to YOLO format
├── train_yolov8_colab.ipynb    # Colab training notebook
├── prepare_for_colab.py        # Create dataset zip for Colab
├── detect_realtime.py          # Real-time detection + Google Maps
├── api_server.py               # FastAPI production server
├── requirements_yolo.txt       # Python dependencies
├── YOLO_QUICKSTART.md         # Detailed setup guide
├── PRODUCTION_STRATEGY.md     # Architecture analysis
└── data/                      # Dataset (train/val folders)
```

## Model Performance

| Model    | Size | Speed (GPU) | mAP@0.5 | Use Case |
|----------|------|-------------|---------|----------|
| YOLOv8n  | 3.2M | 100 FPS     | 88-90%  | Real-time, Mobile |
| YOLOv8s  | 11M  | 80 FPS      | 90-92%  | Balanced |
| YOLOv8m  | 26M  | 60 FPS      | 92-94%  | High Accuracy |
| YOLOv8l  | 44M  | 40 FPS      | 93-95%  | Maximum Accuracy |

## Road Condition Scoring

The system analyzes detected damage and assigns a score (0-100):

- **90-100**: Excellent - No damage
- **75-89**: Good - Minor cracks
- **50-74**: Fair - Multiple cracks or small potholes
- **25-49**: Poor - Major damage, large potholes
- **0-24**: Critical - Severe damage, immediate repair needed

**Damage Weights**:
- Crack: -5 points
- Pothole: -15 points
- Severe Damage: -30 points

# View statistics
python quick_upload.py --stats
```

### Option 3: Manual Organization
```
data/
├── train/
│   ├── Good/
│   ├── Minor_Damage/
│   ├── Pothole/
│   ├── Crack/
│   └── Severe_Damage/
└── validation/
    ├── Good/
    ├── Minor_Damage/
    ├── Pothole/
    ├── Crack/
    └── Severe_Damage/
```

**Requirements:**
- 100-500 images per class
- JPG/PNG format
- Clear road condition photos

## Project Structure

```
├── train_road_model.ipynb    # Jupyter notebook for Colab training
├── prepare_dataset.py        # GUI tool for uploading images
├── quick_upload.py           # Command-line image upload
├── config.py                 # Configuration settings
├── road_condition_model.py   # PyTorch CNN model
├── camera_capture.py         # Camera interface
├── google_maps_integration.py # Google Maps API
├── main.py                   # Main application
├── deploy_model.py           # Model deployment automation
├── verify_model.py           # Model verification tool
├── run_service.py            # Background monitoring service
└── requirements.txt          # Python dependencies
```

## Configuration

Edit `config.py` to customize:

- **API Key**: `GOOGLE_MAPS_API_KEY`
- **Camera**: `CAMERA_INDEX`, `CAMERA_RESOLUTION`
- **Detection**: `CONFIDENCE_THRESHOLD`, `CAPTURE_INTERVAL`
- **GPS**: `USE_MOCK_GPS`, `MOCK_GPS_LOCATION`

## Road Conditions Detected

| Class | Severity | Description |
|-------|----------|-------------|
| Good | None | Road in good condition |
| Minor Damage | Low | Small wear and tear |
| Crack | Medium | Visible cracks in road |
| Pothole | High | Pothole detected |
| Severe Damage | Critical | Major road damage |

## Keyboard Controls (GUI Mode)

- **Q**: Quit application
- **S**: Save current frame
- **R**: Force report current condition

## Training on Google Colab

**Step-by-step workflow:**

1. **Open Notebook**: Open `train_road_model.ipynb` in VS Code
2. **Connect to Colab**: Select kernel → "Connect to Colab"
3. **Enable GPU**: Runtime → Change runtime type → T4 GPU
4. **Upload Dataset**: 
   - Option A: Upload ZIP file directly
   - Option B: Mount Google Drive with dataset
5. **Train**: Run all cells (30-90 minutes on GPU)
6. **Download**: Download `road_condition_model_best.pth`
7. **Deploy**: Run `python deploy_model.py` locally

## Troubleshooting

**Model not found?**
```powershell
python verify_model.py
```

**Camera issues?**
- Check `CAMERA_INDEX` in config.py
- Try different camera indices (0, 1, 2)

**Low confidence predictions?**
- Collect more training data
- Increase training epochs
- Adjust `CONFIDENCE_THRESHOLD`

**Google Maps not working?**
- Verify API key in config.py
- Enable required APIs in Google Cloud Console
- Check `USE_MOCK_GPS` setting

## Requirements

- Python 3.8+
- Webcam or camera
- Internet connection (for Google Maps)
- Google Colab account (for GPU training)

## License

MIT License - feel free to modify and distribute.

## Support

For issues or questions:
1. Check model with `python verify_model.py`
2. Review logs in `logs/` directory
3. Test on single image first
4. Verify all dependencies installed

---

**Built with PyTorch, OpenCV, and Google Maps API**
