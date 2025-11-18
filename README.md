# Road Condition Detection System

AI-powered system to detect road conditions (potholes, cracks, damage) using camera feed and report to Google Maps.

## Features

- **Real-time Detection**: Live camera feed analysis
- **YOLO-based CNN**: Transfer learning with MobileNetV2 backbone
- **Google Maps Integration**: Automated road condition reporting
- **GPU Training**: Train on Google Colab for free GPU access
- **Multiple Modes**: GUI monitoring, background service, single image detection

## Quick Start

### Step 1: Install Dependencies

```powershell
# Install Python packages
pip install -r requirements.txt
```

### Step 2: Train the Model

**Use Google Colab with Jupyter Notebook (GPU - Recommended)**

1. Open `train_road_model.ipynb` in VS Code
2. Click "Open in Colab" or "Select Kernel" → "Connect to Colab"
3. Upload your dataset (ZIP file or Google Drive)
4. Enable GPU runtime (Runtime → Change runtime type → GPU)
5. Run all cells to train
6. Download trained model when complete

**Alternative: Train Locally (CPU - Slow)**
```powershell
# Install Jupyter
pip install jupyter

# Run notebook locally
jupyter notebook train_road_model.ipynb
```

### Step 3: Deploy Trained Model

```powershell
# Copy downloaded model and verify installation
python deploy_model.py
```

### Step 4: Configure Google Maps API

1. Get API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Edit `config.py` and set `GOOGLE_MAPS_API_KEY`

### Step 5: Run Detection

```powershell
# Live monitoring with GUI
python main.py

# Background service (no GUI)
python run_service.py

# Single image detection
python main.py --image path/to/image.jpg

# Verify model
python verify_model.py --image test.jpg
```

## Dataset Preparation

### Option 1: GUI Tool (Easiest)
```powershell
python prepare_dataset.py
```
- Upload images through user-friendly interface
- Automatic train/validation split
- View statistics in real-time
- Export as ZIP for Colab

### Option 2: Command Line
```powershell
# Upload specific images
python quick_upload.py --class Pothole --images img1.jpg img2.jpg img3.jpg

# Upload entire folder
python quick_upload.py --class Crack --folder C:\road_photos\cracks

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
