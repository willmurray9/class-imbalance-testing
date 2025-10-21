# NeMo Data Designer for Stroke Prediction Class Imbalance

## Overview

This guide explains how to apply NeMo Data Designer (NDD) to your stroke prediction class imbalance problem.

## Your Current Situation

**Dataset**: Stroke prediction with severe class imbalance
- Total records: 5,110
- Minority class (stroke=1): 249 (~5%)
- Majority class (stroke=0): 4,861 (~95%)

**Features** (15 total):
- **Numerical** (scaled): `age`, `avg_glucose_level`, `bmi`
- **Binary**: `gender_Male`, `hypertension_Yes`, `heart_disease_Yes`, `ever_married_Yes`, `Residence_type_Urban`
- **Categorical (one-hot)**: 
  - Work type: `work_type_Never_worked`, `work_type_Private`, `work_type_Self-employed`, `work_type_children`
  - Smoking: `smoking_status_formerly smoked`, `smoking_status_never smoked`, `smoking_status_smokes`

**Techniques tested so far**:
1. ✅ Base XGBoost model
2. ✅ Scale positive weight
3. ✅ SMOTE (Synthetic Minority Over-sampling Technique)
4. 🔄 NeMo Data Designer (current)

## Critical Assessment: Is NDD the Right Tool?

### NDD's Strengths
- **Text generation**: Product names, reviews, narratives, descriptions
- **Semantic understanding**: Using LLMs to ensure coherent combinations
- **Categorical reasoning**: Making smart choices for interdependent categorical features

### NDD's Limitations for Your Use Case
- **Your data is numerical/medical**, not text-based
- **SMOTE is designed specifically for numerical feature interpolation**
- **NDD is slower** (API calls) and **more expensive** (token usage)
- **Statistical methods** (like SMOTE) typically perform better on purely numerical data

### Why Test NDD Anyway?

Even though NDD may not be ideal, it's worth testing because:
1. **Learning opportunity**: Understand how LLM-based synthetic data generation works
2. **Medical coherence**: NDD can validate that feature combinations make medical sense
3. **Benchmarking**: Compare LLM approach vs. statistical approach
4. **Future applications**: If you ever add text features (clinical notes, patient histories), NDD becomes more valuable

## Implementation Strategy

### Approach: Minority Class Augmentation with Medical Coherence Validation

Generate synthetic stroke patients (minority class) using:

1. **Samplers for features** - Based on stroke patient distributions
2. **LLM for validation** - Ensure medically plausible combinations
3. **Combine with original data** - Train model on augmented dataset

### Step-by-Step Process

#### Step 1: Analyze Stroke Patient Distributions

```python
stroke_cases = df[df['stroke'] == 1]

# Understand numerical feature distributions
print(stroke_cases[['age', 'avg_glucose_level', 'bmi']].describe())

# Understand categorical feature distributions
for col in ['gender_Male', 'hypertension_Yes', 'heart_disease_Yes', ...]:
    print(f"{col}: {stroke_cases[col].value_counts(normalize=True)}")
```

#### Step 2: Configure NDD Samplers

Use samplers to match the distributions of stroke patients:

```python
from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
    SamplerType,
    UniformSamplerParams,
    CategorySamplerParams,
)

config_builder = DataDesignerConfigBuilder()

# Numerical features - sample from stroke patient ranges
config_builder.add_column(
    SamplerColumnConfig(
        name="age",
        sampler_type=SamplerType.UNIFORM,
        params=UniformSamplerParams(
            low=float(stroke_cases['age'].min()),
            high=float(stroke_cases['age'].max())
        ),
    )
)

# Binary categorical - weighted by stroke patient distribution
gender_dist = stroke_cases['gender_Male'].value_counts(normalize=True)
config_builder.add_column(
    SamplerColumnConfig(
        name="gender_Male",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=[0, 1],
            weights=[gender_dist.get(0, 0.5), gender_dist.get(1, 0.5)]
        ),
        convert_to="int"
    )
)

# Multi-class categorical (work_type, smoking_status)
config_builder.add_column(
    SamplerColumnConfig(
        name="work_type",
        sampler_type=SamplerType.CATEGORY,
        params=CategorySamplerParams(
            values=["Never_worked", "Private", "Self-employed", "children"],
            weights=[
                stroke_cases['work_type_Never_worked'].sum() / len(stroke_cases),
                stroke_cases['work_type_Private'].sum() / len(stroke_cases),
                stroke_cases['work_type_Self-employed'].sum() / len(stroke_cases),
                stroke_cases['work_type_children'].sum() / len(stroke_cases),
            ]
        ),
    )
)
```

#### Step 3: Add LLM Medical Coherence Check (Optional)

This is NDD's unique value-add - using an LLM to validate combinations:

```python
from nemo_microservices.data_designer.essentials import LLMTextColumnConfig

config_builder.add_column(
    LLMTextColumnConfig(
        name="medical_plausibility",
        prompt=(
            "Given a patient with:\\n"
            "- Age: {{ age }}\\n"
            "- Glucose: {{ avg_glucose_level }}\\n"
            "- BMI: {{ bmi }}\\n"
            "- Hypertension: {{ hypertension_Yes }}\\n"
            "- Heart disease: {{ heart_disease_Yes }}\\n"
            "- Smoking: {{ smoking_status }}\\n\\n"
            "Rate medical plausibility for a stroke patient (1-10). "
            "Respond with ONLY a number."
        ),
        model_alias="nemotron-super",  # Best quality
    )
)
```

You can then filter generated records to only keep those with high plausibility scores (e.g., ≥7).

#### Step 4: Generate Synthetic Data

```python
# Start with a preview
preview = data_designer_client.preview(config_builder, num_records=10)

# When ready, generate full dataset
num_synthetic = len(stroke_cases)  # Conservative: 2x minority class
result = data_designer_client.generate(config_builder, num_records=num_synthetic)
synthetic_df = result.dataset
```

#### Step 5: Process and Convert Format

```python
def process_ndd_output(generated_df):
    """Convert NDD output to match your dataset format"""
    processed = pd.DataFrame()
    
    # Copy numerical features directly
    processed['age'] = generated_df['age']
    processed['avg_glucose_level'] = generated_df['avg_glucose_level']
    processed['bmi'] = generated_df['bmi']
    
    # Copy binary features
    processed['gender_Male'] = generated_df['gender_Male'].astype(int)
    processed['hypertension_Yes'] = generated_df['hypertension_Yes'].astype(int)
    # ... etc
    
    # One-hot encode multi-class features
    processed['work_type_Private'] = (generated_df['work_type'] == 'Private').astype(int)
    # ... etc
    
    # All are stroke cases
    processed['stroke'] = 1
    
    return processed

processed_synthetic = process_ndd_output(synthetic_df)
```

#### Step 6: Combine and Train

```python
# Combine original + synthetic
df_original = df.drop('id', axis=1)
df_augmented = pd.concat([df_original, processed_synthetic], ignore_index=True)

# Train model
X = df_augmented.drop('stroke', axis=1)
y = df_augmented['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = xgb.XGBClassifier(tree_method='hist', eval_metric='aucpr')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
# ... calculate metrics
```

## Model Selection: Which NDD Model to Use?

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `nemotron-nano-v2` | 9B | Fast | Good | Prototyping/testing |
| **`nemotron-super`** | **49B** | **Medium** | **Excellent** | **Production (RECOMMENDED)** |
| `mistral-small` | 24B | Medium | Good | Balanced option |
| `gpt-oss-120b` | 120B | Slow | Best | Max quality (expensive) |
| `llama-4-scout-17b` | 17B | Medium | Good | Alternative |

**Recommendation**: Use **`nemotron-super`** for the best balance of quality and speed for your medical coherence validation.

## Expected Results

### What to Expect
- **Performance**: NDD may perform similarly to SMOTE, possibly slightly worse
- **Time**: Much slower than SMOTE (minutes vs. seconds)
- **Cost**: Consumes API credits ($)
- **Learning**: You'll understand LLM-based synthetic data generation

### When NDD Would Excel
- If you had **text features** (patient narratives, clinical notes)
- If features had **complex semantic relationships**
- For **categorical features** where domain knowledge helps
- When you need **explainable/realistic** synthetic samples

### For Your Numerical Medical Data
- SMOTE is more appropriate (designed for numerical interpolation)
- NDD's main advantage is the medical coherence validation
- Statistical methods typically win for pure numerical data

## Augmentation Strategy Options

1. **Conservative** (recommended for testing): 2x minority class
   - Generate: 249 synthetic samples
   - New balance: ~8.8% stroke vs. 91.2% no stroke

2. **Moderate**: 4x minority class
   - Generate: 747 synthetic samples  
   - New balance: ~16.7% stroke vs. 83.3% no stroke

3. **Aggressive**: Full balance
   - Generate: 4,612 synthetic samples
   - New balance: 50% stroke vs. 50% no stroke

Start conservative and increase if results are promising.

## Implementation Checklist

- [ ] Analyze stroke patient feature distributions
- [ ] Configure NDD samplers based on distributions
- [ ] (Optional) Add LLM medical coherence validation
- [ ] Generate small preview (10 samples) to test
- [ ] Verify generated data looks realistic
- [ ] Generate full synthetic dataset
- [ ] Process and convert to match dataset format
- [ ] (Optional) Filter by plausibility score if using LLM validation
- [ ] Combine with original data
- [ ] Train XGBoost model
- [ ] Evaluate and compare with SMOTE/other techniques
- [ ] Document findings

## Cost Considerations

NDD uses NVIDIA API credits. Approximate costs:
- **Preview** (10 samples): Minimal (~$0.01)
- **Conservative** (249 samples): ~$0.50-2.00
- **Full balance** (4,612 samples): ~$10-40

Start with preview and conservative approaches to control costs.

## Final Recommendation

### For Learning: ✅ Do It!
Great way to understand LLM-based synthetic data generation.

### For Production: ⚠️ Probably Not
For your numerical medical data, SMOTE or other statistical methods are:
- Faster
- Cheaper  
- Likely to perform as well or better

### Best Use of NDD
If you ever extend this project to include:
- Patient clinical notes
- Medical narratives
- Symptom descriptions
- Doctor observations

Then NDD would be extremely valuable!

### Pragmatic Approach
1. Complete the NDD implementation (learning experience)
2. Compare results honestly with SMOTE
3. Document that SMOTE is more appropriate for this use case
4. Keep NDD knowledge for future text-based medical datasets

---

**Next Step**: I can create a starter notebook with all the code structured and ready to run. Would you like me to do that?

