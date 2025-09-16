# 📊 TruthGuard AI - Dataset Documentation

## Dataset Overview

This directory contains the training and testing datasets used for TruthGuard AI's misinformation detection models.

### Files Structure

```
data/
├── README.md           # This documentation file
├── train.csv          # Training dataset (labeled news articles)
├── test.csv           # Testing dataset (unlabeled news articles)
└── data_info.json     # Dataset metadata and statistics
```

### Dataset Description

#### Training Dataset (`train.csv`)
- **Purpose**: Training the AI models for misinformation detection
- **Format**: CSV with columns: `id`, `title`, `author`, `text`, `label`
- **Labels**: 
  - `0` = Real news
  - `1` = Fake news
- **Source**: Curated collection of verified real and fake news articles
- **Size**: Multiple thousand articles covering various topics

#### Test Dataset (`test.csv`)
- **Purpose**: Evaluating model performance and judge demonstrations
- **Format**: CSV with columns: `id`, `title`, `author`, `text`
- **Labels**: Not included (for testing purposes)
- **Use Case**: Judge evaluation and performance benchmarking

### Data Quality

#### Content Coverage
- **Political News**: Election coverage, government announcements
- **Health Information**: COVID-19, medical treatments, public health
- **Technology**: Social media, tech company news, innovation
- **Social Issues**: Community events, social movements
- **International News**: Global events and foreign policy

#### Verification Standards
- **Real News**: Sourced from reputable news organizations
- **Fake News**: Verified misinformation from fact-checking organizations
- **Quality Control**: Manual review and verification process
- **Bias Mitigation**: Balanced representation across topics and viewpoints

### Usage Guidelines

#### For Training
```python
import pandas as pd

# Load training data
train_df = pd.read_csv('data/train.csv')

# Prepare features and labels
X = train_df['text']  # or combine title + text
y = train_df['label']

# Split for validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### For Testing
```python
# Load test data
test_df = pd.read_csv('data/test.csv')
X_test = test_df['text']

# Make predictions
predictions = model.predict(X_test)
```

### Dataset Statistics

#### Label Distribution (Training Set)
- **Real News**: ~50% of articles
- **Fake News**: ~50% of articles
- **Total Articles**: Thousands of verified examples

#### Text Characteristics
- **Average Article Length**: 200-2000 words
- **Language**: English
- **Time Period**: Recent years (2016-2024)
- **Geographic Coverage**: Primarily US/International news

### Data Privacy and Ethics

#### Privacy Compliance
- **No Personal Data**: Only published news content included
- **Public Domain**: All articles from publicly available sources
- **Attribution**: Original sources preserved where appropriate
- **Educational Use**: Licensed for research and educational purposes

#### Ethical Considerations
- **Balanced Representation**: Equal representation of real vs fake news
- **No Harmful Content**: Filtered to remove extremely offensive material
- **Context Preservation**: Maintains journalistic context and nuance
- **Fact-Checking Verified**: All labels verified through multiple sources

### Data Preprocessing

#### Cleaning Steps Applied
1. **Text Normalization**: Consistent encoding and formatting
2. **Duplicate Removal**: Eliminated duplicate articles
3. **Quality Filtering**: Removed incomplete or corrupted entries
4. **Label Verification**: Cross-referenced with fact-checking databases

#### Preprocessing Code
```python
def preprocess_text(text):
    """Preprocessing pipeline for news articles"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Handle encoding issues
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Basic cleaning while preserving meaning
    return text.strip()
```

### Model Training Recommendations

#### Feature Engineering
- **Text Features**: TF-IDF, word embeddings, linguistic features
- **Metadata Features**: Author information, publication patterns
- **Structural Features**: Article length, paragraph count, formatting

#### Validation Strategy
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Temporal Split**: If timestamps available, use time-based splits
- **Stratified Sampling**: Maintain label balance in splits

#### Performance Metrics
- **Primary**: Accuracy, Precision, Recall, F1-Score
- **Secondary**: AUC-ROC, Confusion Matrix Analysis
- **Interpretability**: Feature importance, SHAP values

### Dataset Limitations

#### Known Limitations
- **Language**: English only (multi-language support planned)
- **Domain**: Primarily news articles (social media content limited)
- **Time Period**: May not capture most recent misinformation trends
- **Geographic Bias**: Primarily US/Western news sources

#### Mitigation Strategies
- **Regular Updates**: Periodic dataset refreshes planned
- **Domain Expansion**: Adding social media and messaging content
- **Geographic Diversity**: Including more international sources
- **Language Support**: Multi-language datasets in development

### Citation and Attribution

If using this dataset for research, please cite:

```bibtex
@dataset{truthguard_dataset_2024,
  title = {TruthGuard AI Misinformation Detection Dataset},
  author = {TruthGuard AI Team},
  year = {2024},
  publisher = {TruthGuard AI Project},
  note = {Curated dataset for fake news detection research}
}
```

### Contributing to Dataset

#### How to Contribute
- **Submit New Examples**: Provide verified real/fake news examples
- **Quality Review**: Help verify labels and improve accuracy
- **Domain Expansion**: Contribute specialized domain content
- **Multi-language**: Provide non-English content with verified labels

#### Contribution Guidelines
1. **Verification Required**: All submissions must be fact-checked
2. **Source Attribution**: Provide original source information
3. **Quality Standards**: Meet content and formatting requirements
4. **Legal Compliance**: Ensure appropriate licensing and permissions

---

For questions about the dataset or to contribute new examples, please open an issue in the project repository or contact the development team.

**Last Updated**: September 2024
**Dataset Version**: 1.0.0
**License**: Educational and Research Use