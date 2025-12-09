# Dependencies Documentation

## System Requirements

- **Python**: Version 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Tesseract OCR**: System-level installation required

## Installing Tesseract OCR

### macOS
```bash
brew install tesseract
```

### Ubuntu/Debian Linux
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add Tesseract to your system PATH

### Verify Installation
```bash
tesseract --version
```

## Python Dependencies

### Installation Steps

1. **Create Virtual Environment** (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Requirements**
```bash
pip install -r requirements.txt
```

3. **Download spaCy Model**
```bash
python -m spacy download en_core_web_sm
```

### Dependency List and Purpose

#### Core Image Processing
- **opencv-python** (>=4.8.0)
  - Purpose: Image preprocessing (rotation, denoising, contrast enhancement)
  - Used for: Deskewing, bilateral filtering, CLAHE, binarization

- **Pillow** (>=10.0.0)
  - Purpose: Image loading and manipulation
  - Used for: PIL Image format compatibility with Tesseract

- **numpy** (>=1.24.0)
  - Purpose: Numerical array operations
  - Used for: Image array manipulations, statistical calculations

#### OCR
- **pytesseract** (>=0.3.10)
  - Purpose: Python wrapper for Tesseract OCR
  - Used for: Text extraction from images with bounding boxes

#### NLP and PII Detection
- **spacy** (>=3.6.0)
  - Purpose: Natural Language Processing and Named Entity Recognition
  - Used for: Detecting person names, dates, organizations, locations

- **en_core_web_sm**
  - Purpose: English language model for spaCy
  - Used for: NER model for entity detection

#### Text Processing
- **textblob** (>=0.17.1)
  - Purpose: Text processing and spell checking
  - Used for: Optional spell correction in text cleaning

#### Data Handling
- **pandas** (>=2.0.0)
  - Purpose: Data manipulation and analysis
  - Used for: Creating summary statistics and CSV exports

#### Development and Visualization
- **jupyter** (>=1.0.0)
  - Purpose: Jupyter Notebook interface
  - Used for: Interactive pipeline demonstration

- **ipykernel** (>=6.25.0)
  - Purpose: Jupyter kernel for Python
  - Used for: Running Python code in notebooks

- **matplotlib** (>=3.7.0)
  - Purpose: Data visualization and plotting
  - Used for: Visualizing preprocessing steps and PII statistics

#### Utilities
- **tqdm** (>=4.65.0)
  - Purpose: Progress bars
  - Used for: Showing processing progress for multiple images

## Troubleshooting

### Tesseract Not Found
**Error**: `pytesseract.pytesseract.TesseractNotFoundError`

**Solution**:
- Verify Tesseract is installed: `tesseract --version`
- If on Windows, ensure Tesseract is in PATH or set manually:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

### spaCy Model Not Found
**Error**: `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### OpenCV Import Error
**Error**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution** (Ubuntu/Linux):
```bash
sudo apt-get install libgl1-mesa-glx
```

### Low OCR Accuracy
**Issue**: Poor text extraction quality

**Solutions**:
1. Ensure Tesseract is latest version (5.0+)
2. Try different PSM modes in `ocr_engine.py`
3. Adjust preprocessing parameters for your specific images
4. Consider training custom Tesseract models for specific handwriting styles

## Version Compatibility

| Dependency | Tested Version | Minimum Version |
|------------|---------------|-----------------|
| Python | 3.11.5 | 3.8.0 |
| Tesseract | 5.3.3 | 4.0.0 |
| opencv-python | 4.8.1 | 4.8.0 |
| pytesseract | 0.3.10 | 0.3.10 |
| spacy | 3.7.2 | 3.6.0 |
| numpy | 1.26.2 | 1.24.0 |
| Pillow | 10.1.0 | 10.0.0 |

## Optional Dependencies

For improved performance or additional features:

- **opencv-contrib-python**: Additional OpenCV modules
- **textblob-en**: Enhanced English language support for TextBlob
- **pandas-profiling**: Detailed data analysis reports

## Hardware Recommendations

- **RAM**: Minimum 4GB, Recommended 8GB+
- **CPU**: Multi-core processor recommended for batch processing
- **Disk Space**: ~500MB for dependencies, additional space for outputs

## Environment Variables

Optional environment variables for configuration:

```bash
# Tesseract executable path (if not in PATH)
export TESSERACT_CMD=/usr/local/bin/tesseract

# spaCy model name (if using different model)
export SPACY_MODEL=en_core_web_sm
```
