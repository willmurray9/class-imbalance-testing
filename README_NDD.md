# Using NeMo Data Designer for Stroke Prediction

## Quick Start

I've created two resources to help you apply NeMo Data Designer to your stroke prediction class imbalance problem:

### 1. 📚 **NDD_IMPLEMENTATION_GUIDE.md**
Comprehensive guide explaining:
- Why NDD may not be ideal for your numerical medical data (but worth trying as a learning exercise)
- Detailed strategy for applying NDD to stroke prediction
- Step-by-step implementation approach
- Model selection guidance
- Expected results and limitations

### 2. 🐍 **ndd_stroke_augmentation.py**
Ready-to-run Python script that:
- Analyzes your stroke patient distributions
- Configures NDD samplers automatically
- Adds LLM-based medical coherence validation
- Generates synthetic stroke patients
- Combines with original data
- Trains and evaluates XGBoost model
- Reports metrics for comparison

## Running the Script

```bash
# Make sure you're in your virtual environment
# Your .env file should have NVIDIA_API_KEY set

python ndd_stroke_augmentation.py
```

The script will:
1. Load your stroke data
2. Analyze stroke patient distributions
3. Configure NDD samplers
4. Generate a preview (10 samples) for validation
5. Generate full synthetic dataset (default: conservative 2x minority class)
6. Process and combine with original data
7. Train XGBoost model
8. Report comprehensive metrics

## What to Expect

### ⏱️ Time
- **Preview (10 samples)**: ~1-2 minutes
- **Full generation (249 samples)**: ~10-20 minutes

### 💰 Cost
- **Preview**: ~$0.01
- **Conservative (249 samples)**: ~$0.50-2.00
- **Full balance (4,612 samples)**: ~$10-40

### 📊 Performance
Realistically, NDD will likely perform **similar to or slightly worse than SMOTE** for this use case because:
- Your data is primarily numerical (age, glucose, BMI)
- SMOTE is specifically designed for numerical feature interpolation
- LLMs aren't optimized for pure numerical synthesis

**However**, you'll gain:
- Medical coherence validation (unique NDD feature)
- Experience with LLM-based synthetic data generation
- Insights for future text-based medical projects

## Key Configuration Options

### In `ndd_stroke_augmentation.py`

**Model Selection** (line 23):
```python
augmenter = NDDStrokeAugmenter(model_alias="nemotron-super")
# Options: "nemotron-nano-v2", "nemotron-super", "mistral-small", "gpt-oss-120b"
```

**LLM Validation** (line 52):
```python
augmenter.configure_samplers(use_llm_validation=True)  # Set False to skip
```

**Plausibility Filtering** (line 391):
```python
processed_synthetic = augmenter.process_generated_data(
    synthetic_df,
    min_plausibility=7  # Only keep samples with score ≥7, or None for no filtering
)
```

**Generation Amount** (line 406):
```python
num_to_generate = num_minority  # Conservative: 2x minority class
# Or: num_minority * 3          # Moderate: 4x minority class
# Or: num_majority - num_minority  # Aggressive: full balance
```

## Understanding the Output

The script will output:

### 1. Data Analysis
```
📊 Loading data from data/stroke_data_prepared.csv...
   Total records: 5110
   Stroke cases: 249 (4.9%)
   Non-stroke cases: 4861 (95.1%)
```

### 2. Feature Statistics
```
📈 Stroke patient feature statistics:
   Age: 0.684 ± 0.236
   Glucose: 2.345 ± 1.123
   BMI: 0.156 ± 0.421
```

### 3. Generation Progress
```
🏭 Generating 249 synthetic stroke patients...
   ⚠️  This will take several minutes and consume API credits!
   Model: nemotron-super
```

### 4. Final Metrics
```
==============================================================
NeMo Data Designer Model Performance
==============================================================
Accuracy:  0.9456
Precision: 0.3214
Recall:    0.7892
F1 Score:  0.4567
ROC-AUC:   0.8723
PR-AUC:    0.4123
==============================================================
```

## Comparison Framework

Compare your NDD results with other techniques:

| Technique | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-----------|----------|-----------|--------|----|---------|----|
| Base Model | ? | ? | ? | ? | ? | ? |
| Scale Pos Weight | ? | ? | ? | ? | ? | ? |
| SMOTE | ? | ? | ? | ? | ? | ? |
| **NDD** | ? | ? | ? | ? | ? | ? |

Fill in your actual values to compare!

## Troubleshooting

### Error: "Module not found: nemo_microservices.data_designer"
✅ Already fixed! You upgraded to version 1.3.0 which has the stable API.

### Error: "API key not found"
Check your `.env` file has:
```
NVIDIA_API_KEY=your_key_here
```

### Generation is too slow
Try:
- Use `model_alias="nemotron-nano-v2"` (faster, lower quality)
- Set `use_llm_validation=False` (skips LLM coherence check)
- Reduce `num_to_generate`

### Out of API credits
- Check your NVIDIA account balance
- Start with preview only to test
- Use conservative generation amounts

## Realistic Assessment

### ✅ Good Reasons to Use NDD Here
1. **Learning experience** - Understand LLM-based synthesis
2. **Medical coherence** - Validate feature combinations make sense
3. **Completeness** - Compare all available techniques
4. **Future applications** - Practice for text-based medical data

### ⚠️ Limitations for This Use Case
1. **Not optimized for numerical data** - SMOTE is better suited
2. **Slower** - Minutes vs. seconds
3. **Costs money** - API credits
4. **Likely similar performance** - May not beat SMOTE

### 🎯 When NDD Shines
- Patient clinical notes
- Medical narratives
- Symptom descriptions
- Text-based features
- Complex categorical relationships

## Next Steps

1. **Run the preview** to verify everything works
2. **Generate conservative dataset** (249 samples)
3. **Compare metrics** with SMOTE and other techniques
4. **Document findings** honestly:
   - Is NDD worth the time/cost for this use case?
   - What did you learn?
   - When would NDD be more appropriate?
5. **Keep this knowledge** for future text-based medical projects!

## Recommended Model Choice

For this task, use **`nemotron-super`** because:
- ✅ Best balance of quality and speed
- ✅ 49B parameters - smart enough for medical reasoning
- ✅ Reasonable API costs
- ✅ Good at following complex instructions

Avoid:
- ❌ `nemotron-nano-v2` - Too small for medical coherence validation
- ❌ `gpt-oss-120b` - Overkill and expensive for this use case

## Final Thoughts

This is a **valuable learning exercise** even if NDD doesn't outperform SMOTE. You'll gain:

1. **Practical LLM experience** - Understand how to use LLMs for synthetic data
2. **Medical AI insights** - See how LLMs can validate medical coherence
3. **Comparative analysis** - Honest assessment of when to use which technique
4. **Future applications** - Ready for text-based medical datasets

The **real value** of this exercise isn't necessarily beating SMOTE—it's understanding when and how to apply different techniques.

Good luck! 🚀

---

**Questions?** Check `NDD_IMPLEMENTATION_GUIDE.md` for more details or review the commented code in `ndd_stroke_augmentation.py`.

