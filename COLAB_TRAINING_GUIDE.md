# 🚀 Quick Start Guide - Training on Google Colab

## Step 1: Prepare Augmented Dataset (LOCAL)

1. Open terminal/PowerShell in project folder:
   ```bash
   cd C:\Users\amitu\Downloads\yolo
   ```

2. Run augmentation (if not already done):
   ```bash
   python augment_simple.py
   ```
   **Result**: Creates 15,930 training images (6x increase)

3. Create ZIP file for upload:
   ```bash
   python create_augmented_zip.py
   ```
   **Result**: Creates `road_dataset_augmented.zip` (~4.5 GB)

## Step 2: Upload to Google Drive

1. Go to https://drive.google.com/
2. Click **New** → **File upload**
3. Select `road_dataset_augmented.zip`
4. Upload to root of **My Drive** (not in any folder)
5. Wait for upload to complete (~10-20 mins depending on internet)

## Step 3: Run Training on Colab

1. Go to https://colab.research.google.com/
2. Click **File** → **Upload notebook**
3. Select `train_finetuned_model.ipynb` from your project
4. Once uploaded, click **Connect** (top right) to get GPU
5. **Run all cells** (Runtime → Run all)
6. When prompted, authorize Google Drive access

### Training Progress:
- **Setup**: ~5 mins (install packages, extract data)
- **Verification**: Dataset will be detected automatically
- **Training**: ~4-5 hours on Tesla T4 GPU
- **Download**: Model and plots downloaded automatically

## Step 4: Deploy Locally (After Training)

1. Move downloaded model:
   ```bash
   move road_finetuned_best.pth models\road_condition_model.pth
   ```

2. Update `config.py`:
   ```python
   NUM_CLASSES = 3
   CLASS_NAMES = ['Crack', 'Pothole', 'Severe_Damage']
   ```

3. Test deployment:
   ```bash
   python deploy_model.py
   ```

4. Run detection:
   ```bash
   python main.py
   ```

## Expected Results

### With Augmented Dataset (Recommended):
- **Training Images**: 15,930
- **Expected Accuracy**: 87-92%
- **Training Time**: 4-5 hours
- **Improvement**: +6-11% over baseline

### With Original Dataset:
- **Training Images**: 2,655
- **Expected Accuracy**: 83-86%
- **Training Time**: 1-2 hours
- **Improvement**: +2-5% over baseline

## Troubleshooting

### "Dataset not found"
- Make sure ZIP is in root of Google Drive (not in a subfolder)
- File name must be exact: `road_dataset_augmented.zip`

### "Out of memory"
- Reduce `BATCH_SIZE` from 32 to 16 in notebook
- Or use original dataset instead

### "Runtime disconnected"
- Colab free tier has time limits (~12 hours)
- Training should complete in 4-5 hours
- If disconnected, training will be lost - restart

### GPU Not Available
- Click Runtime → Change runtime type → GPU
- If no GPU available, try again later

## Tips for Best Results

1. ✅ Use augmented dataset for best accuracy
2. ✅ Let it train completely (don't interrupt)
3. ✅ Monitor the validation accuracy plot
4. ✅ Download model immediately after training
5. ✅ Save training plots for reference

## What to Expect During Training

```
Epoch 1/100
  Train Acc: 45-50%  Val Acc: 40-45%  ← Initial

Epoch 10/100
  Train Acc: 70-75%  Val Acc: 65-70%  ← Learning

Epoch 30/100
  Train Acc: 85-88%  Val Acc: 80-83%  ← Converging

Epoch 50/100
  Train Acc: 90-93%  Val Acc: 87-90%  ← Best performance

Epoch 60+
  Early stopping triggered ✓
  Best Val Acc: 87-92%
```

## After Training

Your model is now trained! You can:
1. Deploy locally for testing
2. Create web app with Streamlit
3. Integrate with Google Maps API
4. Deploy to cloud (Heroku, AWS, etc.)

Happy training! 🎉
