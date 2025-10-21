# NeMo Data Designer Notebook Guide

## ✅ What's Been Created

I've converted the Python script into a fully functional Jupyter notebook:

**📓 Notebook**: `notebooks/nemo-data-designer-model.ipynb`

## 📋 Notebook Structure

The notebook is organized into 12 sections with 43 cells total:

### Section 1-2: Setup (Cells 0-8)
- Load packages and NDD client
- Load and analyze stroke data
- Understand feature distributions for stroke patients

### Section 3: Configuration (Cells 9-18)
- **Cell 12**: Configure numerical features (age, glucose, BMI)
- **Cell 14**: Configure binary categorical features (gender, hypertension, etc.)
- **Cell 16**: Configure multi-class features (work type, smoking status)
- **Cell 18**: Add optional LLM medical coherence validation

### Section 4: Preview (Cells 19-22)
- **Cell 20**: Generate 10-sample preview (~1-2 minutes)
- **Cell 21**: Display sample record
- **Cell 22**: View preview dataset

### Section 5-6: Generation & Processing (Cells 23-28)
- **Cell 24**: Calculate generation options
- **Cell 25**: ⚠️ **COMMENTED** - Generate full dataset (uncomment when ready)
- **Cell 27**: Process generated data with one-hot encoding
- **Cell 28**: ⚠️ **COMMENTED** - Process full synthetic dataset

### Section 7-8: Combine & Train (Cells 29-32)
- **Cell 30**: ⚠️ **COMMENTED** - Combine original + synthetic data
- **Cell 32**: ⚠️ **COMMENTED** - Train XGBoost model

### Section 9-10: Evaluate & Visualize (Cells 33-36)
- **Cell 34**: ⚠️ **COMMENTED** - Calculate all metrics
- **Cell 36**: ⚠️ **COMMENTED** - Plot confusion matrix

### Section 11: Compare (Cells 37-39)
- **Cell 38**: Comparison table (fill in your actual metrics)
- **Cell 39**: ⚠️ **COMMENTED** - Visual comparison charts

### Section 12: Summary (Cells 40-42)
- **Cell 41**: Workflow summary and recommendations
- **Cell 42**: Additional resources and configuration guide

## 🚀 How to Use

### Initial Run (No API Costs)
Run cells **1-22** to:
1. Set up NDD client
2. Analyze your data
3. Configure samplers
4. Generate a **10-sample preview** (~1-2 minutes)

This lets you verify everything works before committing to the full generation.

### Full Generation (API Costs Apply)
When ready, uncomment and run:
1. **Cell 25** - Generate synthetic data (~10-20 min, ~$0.50-2.00)
2. **Cell 28** - Process synthetic data
3. **Cell 30** - Combine datasets
4. **Cell 32** - Train model
5. **Cell 34** - Evaluate model
6. **Cell 36** - Visualize results

### Final Comparison
1. Fill in metrics from your other notebooks in **Cell 38**
2. Uncomment **Cell 39** to visualize comparison

## 🎛️ Key Configuration Options

### Change Model (Cell 3)
```python
model_alias = "nemotron-super"  # Change to: nemotron-nano-v2, mistral-small, gpt-oss-120b
```

### Toggle LLM Validation (Cell 18)
```python
use_llm_validation = True  # Set to False to skip (faster, cheaper)
```

### Adjust Plausibility Filter (Cell 27)
```python
min_plausibility=7  # Only keep samples scoring ≥7 out of 10
```

### Change Generation Amount (Cell 24)
```python
num_to_generate = num_minority  # Conservative: 249 samples
# num_to_generate = num_minority * 3  # Moderate: 747 samples
# num_to_generate = num_majority - num_minority  # Aggressive: 4,612 samples
```

## 💰 Cost Estimates

| Generation Amount | Samples | Time | Cost (approx) |
|-------------------|---------|------|---------------|
| Preview | 10 | 1-2 min | ~$0.01 |
| Conservative | 249 | 10-20 min | $0.50-2.00 |
| Moderate | 747 | 30-60 min | $1.50-6.00 |
| Aggressive | 4,612 | 2-6 hours | $10-40 |

*Costs vary based on model choice and whether LLM validation is enabled*

## ⚠️ Important Notes

1. **Cells are commented by default** after the preview to prevent accidental API usage
2. **Run the preview first** (cells 1-22) to verify everything works
3. **Start conservative** (249 samples) for initial testing
4. **LLM validation is optional** but provides medical coherence checking
5. **Realistic expectations**: NDD may not beat SMOTE for numerical data

## 📊 Expected Results

Based on the nature of your stroke dataset (primarily numerical):
- **NDD performance**: Similar to or slightly below SMOTE
- **NDD advantage**: Medical coherence validation via LLM
- **SMOTE advantage**: Faster, cheaper, designed for numerical data
- **Learning value**: High - understand LLM-based synthetic data generation

## 🎯 Workflow Summary

```
1. Run cells 1-22 → Generate preview (no major costs)
2. Review preview → Verify quality
3. Uncomment cell 25 → Generate full dataset (costs apply)
4. Uncomment cells 28-34 → Process, train, evaluate
5. Fill cell 38 → Add metrics from other notebooks
6. Uncomment cell 39 → Visualize comparison
7. Document findings → When to use NDD vs SMOTE
```

## 📚 Additional Files

- **`NDD_IMPLEMENTATION_GUIDE.md`** - Detailed strategy and approach
- **`README_NDD.md`** - Quick start guide
- **`ndd_stroke_augmentation.py`** - Python script version (if you prefer)

## 🤔 Questions?

- **"Should I use NDD for this?"** - It's worth trying for learning, but SMOTE is likely better suited for your numerical data
- **"Which model should I use?"** - Start with `nemotron-super` (best quality/speed balance)
- **"Should I enable LLM validation?"** - Yes if you want medical coherence checking, no if prioritizing speed/cost
- **"How many samples should I generate?"** - Start conservative (249), increase if results are promising

---

**✓ Notebook is ready to use!** Start by running cells 1-22 to generate your preview.

