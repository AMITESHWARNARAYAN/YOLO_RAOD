# 🚀 Google Colab Training Guide

## Quick Steps

### 1. Upload Files to Colab

1. Go to https://colab.research.google.com/
2. Upload `train_yolov8_colab.ipynb`
3. Upload `yolo_data.zip` (1.7GB) OR save to Google Drive

### 2. Enable GPU

- Click **Runtime** → **Change runtime type** → **Hardware accelerator: GPU (T4)**
- Click **Save**

### 3. Run Training

- Click **Runtime** → **Run all**
- Or run cells one by one (Shift+Enter)
- Training takes ~30-60 minutes with GPU

### 4. Download Trained Model

When training completes:
- A file `trained_model.zip` will be created
- Click download button in Colab file browser
- Or it will auto-download if `files.download()` runs

---

## Expected Output

### Training Progress
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
  1/50     2.5G      1.234     0.567      1.234        100       640
  2/50     2.5G      0.987     0.456      1.123        100       640
  ...
 50/50     2.5G      0.234     0.123      0.456        100       640
```

### Final Metrics
```
mAP@0.5: 0.8934
mAP@0.5:0.95: 0.6234
Precision: 0.8756
Recall: 0.8512

Per-class AP@0.5:
  Crack: 0.8823
  Pothole: 0.9145
  Severe_Damage: 0.8834
```

---

## Files You'll Download

**trained_model.zip** contains:
- `best.pt` - Best model checkpoint (use this!)
- `last.pt` - Last epoch checkpoint
- `best.onnx` - ONNX format for production
- `best.torchscript` - TorchScript format

---

## After Training

### 1. Extract Model Locally

```powershell
# Extract to project folder
Expand-Archive -Path trained_model.zip -DestinationPath .
```

### 2. Test Detection

```powershell
# Update model path in detect_realtime.py (line ~20):
# model = YOLO('runs/detect/road_damage/weights/best.pt')

# Run detection
python detect_realtime.py --source 0
```

### 3. Deploy API

```powershell
# Update model path in api_server.py (line ~15):
# model = YOLO('runs/detect/road_damage/weights/best.pt')

# Start server
python api_server.py
```

---

## Troubleshooting

### ❌ "Not enough GPU memory"
**Solution**: Reduce batch size in training cell:
```python
results = model.train(
    batch=8,  # Change from 16 to 8
    ...
)
```

### ❌ "Dataset not found"
**Solution**: Make sure you extracted `yolo_data.zip`:
```python
!unzip -q yolo_data.zip
```

### ❌ "Training too slow"
**Solution**: Check GPU is enabled:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### ❌ "Low accuracy (<80%)"
**Solutions**:
- Train more epochs (100 instead of 50)
- Use larger model (yolov8s or yolov8m)
- Check dataset quality

### ❌ "Disconnected from runtime"
**Solution**: 
- Colab free tier has time limits (~12 hours)
- Training should complete in 1 hour
- Checkpoints are saved, can resume from last epoch

---

## Model Selection Guide

| Model | Training Time | GPU Memory | Best For |
|-------|---------------|------------|----------|
| yolov8n | 30 min | 2GB | Quick testing, mobile |
| yolov8s | 45 min | 3GB | **Recommended balance** |
| yolov8m | 60 min | 4GB | Higher accuracy |
| yolov8l | 90 min | 6GB | Maximum accuracy |
| yolov8x | 120 min | 8GB | Research (may need Colab Pro) |

**Recommendation**: Start with `yolov8n` for testing, then use `yolov8s` for production.

---

## Advanced: Training on Your Own Images

If you want to add more images:

1. **Add to dataset locally**:
   ```
   data/train/Crack/        (add crack images)
   data/train/Pothole/      (add pothole images)
   data/train/Severe_Damage/ (add severe damage images)
   ```

2. **Reconvert dataset**:
   ```powershell
   python convert_to_yolo_format.py
   python prepare_for_colab.py
   ```

3. **Upload new yolo_data.zip to Colab and retrain**

---

## Tips for Best Results

✅ **Use GPU** - 100x faster than CPU  
✅ **Train 50-100 epochs** - More epochs = better accuracy  
✅ **Use YOLOv8s or YOLOv8m** - Best balance of speed and accuracy  
✅ **Check validation images** - View predicted results in notebook  
✅ **Save to Google Drive** - Don't lose your work if disconnected  
✅ **Export to ONNX** - Faster inference in production  

---

## Need Help?

Check these files:
- `YOLO_QUICKSTART.md` - Complete setup guide
- `PRODUCTION_STRATEGY.md` - Architecture details
- `README.md` - Project overview

Or review Colab outputs - they're very detailed!
