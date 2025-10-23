"""
NeMo Data Designer for Stroke Prediction Class Imbalance
---------------------------------------------------------
Generates synthetic stroke patients (minority class) using NDD
"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score
)
import xgboost as xgb
import re

# Load environment
load_dotenv()

# Import NDD
from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    LLMTextColumnConfig,
    NeMoDataDesignerClient,
    UniformSamplerParams,
    SamplerColumnConfig,
    SamplerType,
)


class NDDStrokeAugmenter:
    """Generate synthetic stroke patients using NeMo Data Designer"""
    
    def __init__(self, model_alias="nemotron-super"):
        """
        Initialize the augmenter
        
        Args:
            model_alias: NDD model to use. Options:
                - "nemotron-nano-v2" (9B, fast)
                - "nemotron-super" (49B, recommended)
                - "mistral-small" (24B)
                - "gpt-oss-120b" (120B, best quality)
        """
        self.model_alias = model_alias
        self.client = NeMoDataDesignerClient(
            base_url="https://ai.api.nvidia.com/v1/nemo/dd",
            default_headers={"Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"}
        )
        self.config_builder = None
        print(f"✓ NDD Client initialized with model: {model_alias}")
    
    def load_and_analyze_data(self, csv_path):
        """Load data and analyze stroke patient distributions"""
        print(f"\n📊 Loading data from {csv_path}...")
        
        self.df = pd.read_csv(csv_path)
        self.stroke_cases = self.df[self.df['stroke'] == 1]
        self.non_stroke_cases = self.df[self.df['stroke'] == 0]
        
        print(f"   Total records: {len(self.df)}")
        print(f"   Stroke cases: {len(self.stroke_cases)} ({len(self.stroke_cases)/len(self.df)*100:.1f}%)")
        print(f"   Non-stroke cases: {len(self.non_stroke_cases)} ({len(self.non_stroke_cases)/len(self.df)*100:.1f}%)")
        
        # Analyze distributions
        print(f"\n📈 Stroke patient feature statistics:")
        print(f"   Age: {self.stroke_cases['age'].mean():.3f} ± {self.stroke_cases['age'].std():.3f}")
        print(f"   Glucose: {self.stroke_cases['avg_glucose_level'].mean():.3f} ± {self.stroke_cases['avg_glucose_level'].std():.3f}")
        print(f"   BMI: {self.stroke_cases['bmi'].mean():.3f} ± {self.stroke_cases['bmi'].std():.3f}")
        
        return self.df, self.stroke_cases
    
    def configure_samplers(self, use_llm_validation=True):
        """
        Configure NDD samplers based on stroke patient distributions
        
        Args:
            use_llm_validation: Whether to add LLM-based medical coherence check
        """
        print(f"\n⚙️  Configuring NDD samplers...")
        
        self.config_builder = DataDesignerConfigBuilder()
        
        # === NUMERICAL FEATURES ===
        # Age (scaled)
        self.config_builder.add_column(
            SamplerColumnConfig(
                name="age",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(
                    low=float(self.stroke_cases['age'].min()),
                    high=float(self.stroke_cases['age'].max())
                ),
            )
        )
        
        # Average glucose level (scaled)
        self.config_builder.add_column(
            SamplerColumnConfig(
                name="avg_glucose_level",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(
                    low=float(self.stroke_cases['avg_glucose_level'].min()),
                    high=float(self.stroke_cases['avg_glucose_level'].max())
                ),
            )
        )
        
        # BMI (scaled)
        self.config_builder.add_column(
            SamplerColumnConfig(
                name="bmi",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(
                    low=float(self.stroke_cases['bmi'].min()),
                    high=float(self.stroke_cases['bmi'].max())
                ),
            )
        )
        
        # === BINARY CATEGORICAL FEATURES ===
        binary_features = [
            'gender_Male', 'hypertension_Yes', 'heart_disease_Yes',
            'ever_married_Yes', 'Residence_type_Urban'
        ]
        
        for feature in binary_features:
            dist = self.stroke_cases[feature].value_counts(normalize=True).to_dict()
            self.config_builder.add_column(
                SamplerColumnConfig(
                    name=feature,
                    sampler_type=SamplerType.CATEGORY,
                    params=CategorySamplerParams(
                        values=[0, 1],
                        weights=[dist.get(0, 0.5), dist.get(1, 0.5)]
                    ),
                    convert_to="int"
                )
            )
        
        # === MULTI-CLASS CATEGORICAL: WORK TYPE ===
        work_types = ["Never_worked", "Private", "Self-employed", "children"]
        work_weights = [
            self.stroke_cases['work_type_Never_worked'].sum() / len(self.stroke_cases),
            self.stroke_cases['work_type_Private'].sum() / len(self.stroke_cases),
            self.stroke_cases['work_type_Self-employed'].sum() / len(self.stroke_cases),
            self.stroke_cases['work_type_children'].sum() / len(self.stroke_cases),
        ]
        
        self.config_builder.add_column(
            SamplerColumnConfig(
                name="work_type",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=work_types,
                    weights=work_weights
                ),
            )
        )
        
        # === MULTI-CLASS CATEGORICAL: SMOKING STATUS ===
        smoking_statuses = ["formerly smoked", "never smoked", "smokes"]
        smoking_weights = [
            self.stroke_cases['smoking_status_formerly smoked'].sum() / len(self.stroke_cases),
            self.stroke_cases['smoking_status_never smoked'].sum() / len(self.stroke_cases),
            self.stroke_cases['smoking_status_smokes'].sum() / len(self.stroke_cases),
        ]
        
        self.config_builder.add_column(
            SamplerColumnConfig(
                name="smoking_status",
                sampler_type=SamplerType.CATEGORY,
                params=CategorySamplerParams(
                    values=smoking_statuses,
                    weights=smoking_weights
                ),
            )
        )
        
        print(f"   ✓ Configured samplers for all features")
        
        # === OPTIONAL: LLM MEDICAL COHERENCE VALIDATION ===
        if use_llm_validation:
            print(f"   ✓ Adding LLM medical coherence validation...")
            self.config_builder.add_column(
                LLMTextColumnConfig(
                    name="medical_plausibility",
                    prompt=(
                        "Given a stroke patient with:\\n"
                        "- Age (scaled): {{ age }}\\n"
                        "- Glucose level (scaled): {{ avg_glucose_level }}\\n"
                        "- BMI (scaled): {{ bmi }}\\n"
                        "- Hypertension: {{ hypertension_Yes }}\\n"
                        "- Heart disease: {{ heart_disease_Yes }}\\n"
                        "- Smoking: {{ smoking_status }}\\n\\n"
                        "Rate the medical plausibility of this combination (1-10). "
                        "Higher scores mean more typical stroke patient profile. "
                        "Respond with ONLY a number from 1 to 10."
                    ),
                    system_prompt=(
                        "You are a medical expert evaluating stroke patient profiles. "
                        "Consider how these risk factors typically occur together."
                    ),
                    model_alias=self.model_alias,
                )
            )
            print(f"   ✓ LLM validation configured")
    
    def generate_preview(self, num_records=10):
        """Generate a preview to verify configuration"""
        print(f"\n🔍 Generating preview ({num_records} samples)...")
        print(f"   This may take a minute...\n")
        
        preview = self.client.preview(self.config_builder, num_records=num_records)
        
        print(f"✓ Preview generated!")
        return preview
    
    def generate_synthetic_data(self, num_records):
        """
        Generate synthetic stroke patients
        
        Args:
            num_records: Number of synthetic samples to generate
        """
        print(f"\n🏭 Generating {num_records} synthetic stroke patients...")
        print(f"   ⚠️  This will take several minutes and consume API credits!")
        print(f"   Model: {self.model_alias}\n")
        
        result = self.client.create(self.config_builder, num_records=num_records)
        
        print(f"\n✓ Generated {len(result.dataset)} synthetic samples!")
        return result.dataset
    
    def process_generated_data(self, generated_df, min_plausibility=None):
        """
        Convert generated data to match original dataset format
        
        Args:
            generated_df: DataFrame from NDD
            min_plausibility: Minimum plausibility score to keep (if using LLM validation)
        """
        print(f"\n🔄 Processing generated data...")
        
        # Filter by plausibility if specified
        if min_plausibility is not None and 'medical_plausibility' in generated_df.columns:
            original_len = len(generated_df)
            # Extract just the numeric value from the plausibility column
            generated_df['plausibility_score'] = generated_df['medical_plausibility'].apply(
                self._extract_plausibility_score
            )
            generated_df = generated_df[
                generated_df['plausibility_score'] >= min_plausibility
            ]
            print(f"   ✓ Extracted plausibility scores (1-10) from LLM responses")
            print(f"   Filtered by plausibility ≥{min_plausibility}: {original_len} → {len(generated_df)}")
        
        processed = pd.DataFrame()
        
        # Numerical features
        processed['age'] = generated_df['age']
        processed['avg_glucose_level'] = generated_df['avg_glucose_level']
        processed['bmi'] = generated_df['bmi']
        
        # Binary features
        processed['gender_Male'] = generated_df['gender_Male'].astype(int)
        processed['hypertension_Yes'] = generated_df['hypertension_Yes'].astype(int)
        processed['heart_disease_Yes'] = generated_df['heart_disease_Yes'].astype(int)
        processed['ever_married_Yes'] = generated_df['ever_married_Yes'].astype(int)
        processed['Residence_type_Urban'] = generated_df['Residence_type_Urban'].astype(int)
        
        # One-hot encode work_type
        processed['work_type_Never_worked'] = (generated_df['work_type'] == 'Never_worked').astype(int)
        processed['work_type_Private'] = (generated_df['work_type'] == 'Private').astype(int)
        processed['work_type_Self-employed'] = (generated_df['work_type'] == 'Self-employed').astype(int)
        processed['work_type_children'] = (generated_df['work_type'] == 'children').astype(int)
        
        # One-hot encode smoking_status
        processed['smoking_status_formerly smoked'] = (generated_df['smoking_status'] == 'formerly smoked').astype(int)
        processed['smoking_status_never smoked'] = (generated_df['smoking_status'] == 'never smoked').astype(int)
        processed['smoking_status_smokes'] = (generated_df['smoking_status'] == 'smokes').astype(int)
        
        # Add stroke label (all synthetic samples are stroke cases)
        processed['stroke'] = 1
        
        print(f"   ✓ Processed {len(processed)} samples")
        return processed
    
    @staticmethod
    def _extract_plausibility_score(text):
        """
        Extract numeric plausibility score (1-10) from LLM response
        Handles responses with <think> tags and explanations
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Numeric score (1-10) or NaN if extraction fails
        """
        if pd.isna(text):
            return np.nan
        
        text = str(text).strip()
        
        # Remove <think>...</think> tags and their content
        text_cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Find all numbers in the remaining text
        numbers = re.findall(r'\b([1-9]|10)\b', text_cleaned)
        
        if numbers:
            # Return the first valid number found
            return int(numbers[0])
        
        # Fallback: try to find any number (1-10) in original text
        numbers = re.findall(r'\b([1-9]|10)\b', text)
        if numbers:
            return int(numbers[0])
        
        return np.nan

    def create_augmented_dataset(self, processed_synthetic_df):
        """Combine original data with synthetic data"""
        print(f"\n🔗 Creating augmented dataset...")
        
        # Remove ID column from original if present
        df_original = self.df.drop('id', axis=1) if 'id' in self.df.columns else self.df.copy()
        
        # Ensure column order matches
        processed_synthetic_df = processed_synthetic_df[df_original.columns]
        
        # Combine
        df_augmented = pd.concat([df_original, processed_synthetic_df], ignore_index=True)
        
        print(f"   Original: {len(df_original)} samples")
        print(f"   Synthetic: {len(processed_synthetic_df)} samples")
        print(f"   Augmented: {len(df_augmented)} samples")
        print(f"\n   Class distribution:")
        print(f"   Stroke: {df_augmented['stroke'].sum()} ({df_augmented['stroke'].sum()/len(df_augmented)*100:.1f}%)")
        print(f"   No stroke: {(df_augmented['stroke']==0).sum()} ({(df_augmented['stroke']==0).sum()/len(df_augmented)*100:.1f}%)")
        
        return df_augmented
    
    def train_and_evaluate(self, df_augmented):
        """Train XGBoost model and evaluate"""
        print(f"\n🤖 Training XGBoost model...")
        
        # Split features and target
        X = df_augmented.drop('stroke', axis=1)
        y = df_augmented['stroke']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = xgb.XGBClassifier(
            tree_method='hist',
            eval_metric='aucpr',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
        }
        
        print(f"\n" + "=" * 60)
        print(f"NeMo Data Designer Model Performance")
        print(f"=" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"=" * 60)
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
        
        return model, metrics


def main():
    """Main workflow"""
    print("=" * 70)
    print("NeMo Data Designer for Stroke Prediction Class Imbalance")
    print("=" * 70)
    
    # Initialize
    augmenter = NDDStrokeAugmenter(model_alias="nemotron-super")
    
    # Load data
    df, stroke_cases = augmenter.load_and_analyze_data('data/stroke_data_prepared.csv')
    
    # Configure samplers
    augmenter.configure_samplers(use_llm_validation=True)
    
    # Preview (test configuration)
    print(f"\n" + "=" * 70)
    print(f"STEP 1: Preview Generation (Testing Configuration)")
    print(f"=" * 70)
    preview = augmenter.generate_preview(num_records=10)
    print(f"\n✓ Preview successful! Configuration looks good.")
    print(f"\nSample record:")
    preview.display_sample_record()
    
    # Ask user if they want to continue
    print(f"\n" + "=" * 70)
    print(f"Ready to generate full synthetic dataset")
    print(f"=" * 70)
    
    # Calculate generation options
    num_minority = len(stroke_cases)
    num_majority = len(df[df['stroke'] == 0])
    
    print(f"\nGeneration options:")
    print(f"  1. Conservative (2x minority): {num_minority} samples")
    print(f"  2. Moderate (4x minority): {num_minority * 3} samples")
    print(f"  3. Aggressive (full balance): {num_majority - num_minority} samples")
    
    # For automation, use conservative
    num_to_generate = num_minority  # Conservative approach
    
    print(f"\nGenerating {num_to_generate} synthetic stroke patients...")
    print(f"⚠️  This will take several minutes!")
    
    # Generate
    synthetic_df = augmenter.generate_synthetic_data(num_to_generate)
    
    # Process
    processed_synthetic = augmenter.process_generated_data(
        synthetic_df,
        min_plausibility=7  # Only keep plausibility ≥7
    )
    
    # Create augmented dataset
    df_augmented = augmenter.create_augmented_dataset(processed_synthetic)
    
    # Train and evaluate
    model, metrics = augmenter.train_and_evaluate(df_augmented)
    
    print(f"\n" + "=" * 70)
    print(f"✓ Complete!")
    print(f"=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Compare these metrics with your SMOTE and scale_pos_weight models")
    print(f"  2. Consider the trade-offs (speed, cost, performance)")
    print(f"  3. Document your findings")
    
    return metrics


if __name__ == "__main__":
    main()

