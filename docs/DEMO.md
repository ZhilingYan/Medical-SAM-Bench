# SAMed2 Interactive Demo with SliceUI

## Overview

SAMed2 provides an interactive web-based demo powered by SliceUI, allowing users to easily segment medical images using our trained models. The demo supports various medical imaging modalities and provides real-time visualization of segmentation results.

## Features

- 🖱️ **Interactive Segmentation**: Click to add positive/negative prompts
- 🏥 **Multi-Modal Support**: Works with CT, MRI, X-Ray, Ultrasound, and more
- 🎯 **Real-time Results**: Instant segmentation feedback
- 📊 **3D Visualization**: Support for volumetric medical data
- 💾 **Export Options**: Save segmentation masks in various formats
- 🔄 **Model Comparison**: Compare different models side-by-side

## Quick Start

### Online Demo

Visit our online demo at: [https://samed2-demo.github.io](https://samed2-demo.github.io)

### Local Installation

1. **Navigate to the demo directory**:
```bash
cd Code/slicerUI
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download model weights**:
```bash
# Download SAMed2 model
wget https://drive.google.com/xxx -O models/samed2_latest.pth
```

4. **Run the demo**:
```bash
python annotation_app.py --model samed2 --port 8080
```

5. **Open in browser**:
Navigate to `http://localhost:8080`

## Usage Guide

### Basic Workflow

1. **Upload Image**: Click "Upload" or drag-and-drop your medical image
2. **Select Model**: Choose from SAMed2, MedSAM2, or other available models
3. **Add Prompts**: 
   - Left-click: Add positive prompt (include this region)
   - Right-click: Add negative prompt (exclude this region)
   - Box prompt: Draw a bounding box around the target
4. **Generate Mask**: Click "Segment" or press Space
5. **Refine**: Add more prompts to improve the segmentation
6. **Export**: Save the final mask using "Export" button

### Supported File Formats

**Input**:
- Images: PNG, JPG, TIFF, DICOM
- Volumes: NIfTI (.nii, .nii.gz), DICOM series

**Output**:
- Masks: PNG, NIfTI, NumPy array
- Visualizations: PNG, SVG

### Advanced Features

#### 3D Volume Segmentation

For 3D medical volumes:

```python
# Load and segment 3D volume
python annotation_app.py --model samed2 --3d-mode --slice-view axial
```

Navigation controls:
- Scroll: Navigate through slices
- Ctrl+Scroll: Zoom in/out
- Shift+Drag: Pan the view

#### Batch Processing

Process multiple images:

```bash
python batch_process.py \
    --input_dir /path/to/images \
    --output_dir /path/to/masks \
    --model samed2 \
    --prompt_mode auto
```

#### Model Comparison

Compare different models:

```bash
python annotation_app.py --compare-mode --models samed2,medsam2,sam2
```

## Demo Video

Watch our demo video to see SAMed2 in action:

[![SAMed2 Demo](https://img.youtube.com/vi/DEMO_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=DEMO_VIDEO_ID)

## API Usage

For programmatic access:

```python
from slicerUI import SAMed2Demo

# Initialize demo
demo = SAMed2Demo(model_path="models/samed2_latest.pth")

# Load image
image = demo.load_image("path/to/medical_image.png")

# Add prompts
demo.add_point_prompt(x=256, y=256, label=1)  # Positive prompt
demo.add_box_prompt(x1=100, y1=100, x2=400, y2=400)

# Generate segmentation
mask = demo.segment()

# Save result
demo.save_mask(mask, "output_mask.png")
```

## Configuration Options

Create a `config.yaml` file:

```yaml
model:
  name: samed2
  checkpoint: models/samed2_latest.pth
  device: cuda
  
interface:
  port: 8080
  host: 0.0.0.0
  debug: false
  
segmentation:
  image_size: 1024
  num_masks: 1
  confidence_threshold: 0.9
  
visualization:
  opacity: 0.5
  colors:
    positive: "#00FF00"
    negative: "#FF0000"
    mask: "#0080FF"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce image size: `--image-size 512`
   - Use CPU mode: `--device cpu`

2. **Slow Performance**:
   - Enable GPU: Ensure CUDA is properly installed
   - Reduce model size: Use SAMed2-T instead of SAMed2-S

3. **Import Errors**:
   - Verify all dependencies: `pip install -r requirements.txt`
   - Check Python version: Requires Python 3.8+

### Performance Tips

- **GPU Acceleration**: Use NVIDIA GPU with CUDA 11.8+
- **Batch Processing**: Process multiple slices simultaneously
- **Memory Management**: Clear cache between large volumes

## Development

### Adding Custom Models

```python
# custom_model.py
from slicerUI.base_model import BaseSegmentationModel

class CustomModel(BaseSegmentationModel):
    def __init__(self, checkpoint_path):
        super().__init__()
        # Load your model
        
    def segment(self, image, prompts):
        # Implement segmentation logic
        return mask
```

### Extending the UI

The demo uses Gradio for the interface. Customize in `annotation_app.py`:

```python
# Add custom UI elements
with gr.Row():
    custom_button = gr.Button("Custom Action")
    custom_slider = gr.Slider(0, 1, 0.5, label="Custom Parameter")
```

## Citation

If you use the SAMed2 demo in your work, please cite:

```bibtex
@software{samed2demo2025,
  title={SAMed2 Interactive Demo},
  author={Yan, Zhiling and others},
  year={2025},
  url={https://github.com/yourusername/SAMed2}
}
```

## Support

For issues or questions:
- GitHub Issues: [https://github.com/yourusername/SAMed2/issues](https://github.com/yourusername/SAMed2/issues)
- Email: samed2@example.com 