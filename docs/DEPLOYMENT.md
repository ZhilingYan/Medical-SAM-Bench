# SAMed2 Website Deployment Guide

## Overview

This guide explains how to deploy the SAMed2 project website, similar to [LLaVA's website](https://llava-vl.github.io/).

## File Structure

```
docs/
├── index.html          # Main HTML file
├── css/
│   └── style.css      # Stylesheet
├── js/
│   └── main.js        # JavaScript for interactions
├── images/            # Image assets
│   ├── favicon.png
│   ├── samed2_overview.png
│   ├── architecture.png
│   ├── results_comparison.png
│   └── demo_preview.png
├── DEMO.md            # Demo documentation
├── MODEL_ZOO.md       # Model zoo documentation
├── MEDBANK.md         # Dataset documentation
└── DEPLOYMENT.md      # This file
```

## Deployment Options

### Option 1: GitHub Pages (Recommended)

1. **Enable GitHub Pages**:
   - Go to your repository settings
   - Navigate to "Pages" section
   - Select source: "Deploy from a branch"
   - Choose branch: `main` (or `gh-pages`)
   - Select folder: `/docs`
   - Click "Save"

2. **Access your site**:
   - Your site will be available at: `https://yourusername.github.io/SAMed2/`

### Option 2: Custom Domain

1. **Add CNAME file**:
   ```bash
   echo "samed2.yourdomain.com" > docs/CNAME
   ```

2. **Configure DNS**:
   - Add a CNAME record pointing to `yourusername.github.io`

### Option 3: Local Development

1. **Using Python HTTP Server**:
   ```bash
   cd docs
   python -m http.server 8000
   ```
   Visit: `http://localhost:8000`

2. **Using Node.js**:
   ```bash
   npx http-server docs -p 8000
   ```

## Required Assets

Before deploying, ensure you have:

1. **Images** (place in `docs/images/`):
   - `samed2_overview.png` - Main hero image showing your method
   - `architecture.png` - Model architecture diagram
   - `results_comparison.png` - Visual comparison results
   - `demo_preview.png` - Screenshot of your demo
   - `favicon.png` - 32x32 or 64x64 icon

2. **Update Links**:
   - Replace placeholder links in `index.html`:
     - ArXiv paper link
     - GitHub repository URL
     - Author homepages
     - Google Drive links for models
     - Demo video ID

## Customization

### Colors
Edit CSS variables in `style.css`:
```css
:root {
    --primary-color: #2563eb;    /* Main blue color */
    --secondary-color: #10b981;   /* Green accent */
    --accent-color: #f59e0b;      /* Orange accent */
}
```

### Content
1. Update author information in `index.html`
2. Replace performance numbers with your actual results
3. Add your funding acknowledgments
4. Update citation with your paper details

### Adding Sections
To add a new section, use this template:
```html
<section class="your-section">
    <div class="container">
        <h2 class="section-title">Your Title</h2>
        <!-- Your content here -->
    </div>
</section>
```

## Performance Optimization

1. **Optimize Images**:
   ```bash
   # Install imageoptim-cli
   npm install -g imageoptim-cli
   
   # Optimize all images
   imageoptim docs/images/*
   ```

2. **Minify CSS/JS** (optional):
   ```bash
   # Install minifiers
   npm install -g cssnano uglify-js
   
   # Minify files
   cssnano docs/css/style.css > docs/css/style.min.css
   uglifyjs docs/js/main.js > docs/js/main.min.js
   ```

## Testing

1. **Check all links**:
   - Paper link works
   - GitHub repository exists
   - Demo is accessible
   - Download links are valid

2. **Test responsiveness**:
   - Mobile devices
   - Tablets
   - Desktop browsers

3. **Browser compatibility**:
   - Chrome
   - Firefox
   - Safari
   - Edge

## Maintenance

- Keep model download links updated
- Update performance numbers as needed
- Add new publications/citations
- Update demo with new features

## Troubleshooting

**Images not loading**:
- Check file paths are relative
- Ensure images are committed to repository

**GitHub Pages not updating**:
- Clear browser cache
- Wait 10 minutes for changes to propagate
- Check GitHub Actions for build errors

**Custom domain not working**:
- Verify CNAME file exists
- Check DNS propagation (can take 24-48 hours)
- Ensure HTTPS is enforced in GitHub settings 