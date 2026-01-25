# 🏀 Unified Basketball Analysis System

A comprehensive basketball analysis platform that combines multiple AI-powered tools for shot analysis, shot detection, and dribble counting.

## 📋 Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [API Integration](#api-integration)
- [Troubleshooting](#troubleshooting)

## ✨ Features

This system integrates **4 complete basketball analysis projects**:

1. **AI Basketball Analysis** (OpenPose)
   - Shot counting (makes and misses)
   - Pose analysis during shooting
   - Elbow and knee angle calculation
   - Release angle and release time
   - Source: https://github.com/hardik0/AI-basketball-analysis

2. **Shot Detection Tracker**
   - Real-time shot detection
   - Basketball tracking
   - Shot trajectory analysis
   - Source: https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker

3. **Dribble Counter (YOLOv8)**
   - Automated dribble counting
   - YOLOv8-based detection
   - Ball trajectory tracking
   - Source: https://github.com/siddhi-lipare/Basketball_Dribbles_Count_Using_YOLOv8

4. **Dribble Counter (Gradio)**
   - Advanced dribble analysis
   - Gradio web interface
   - Real-time processing
   - Source: https://huggingface.co/spaces/gangulya111/basket-bll-dribble

5. **ShotTracker API Integration**
   - Professional basketball analytics
   - Event tracking
   - API: https://developer.shottracker.com/basketball/event.html

## 💻 System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (required for OpenPose)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space

### Software
- **Python**: 3.8 - 3.10
- **CUDA**: 11.7 or higher
- **cuDNN**: Compatible with your CUDA version
- **Git**: Latest version

### Operating System
- Windows 10/11
- Ubuntu 18.04+
- macOS 10.15+ (with Metal support)

## 🚀 Installation

### Step 1: Clone This Repository

```bash
git clone https://github.com/YOUR_USERNAME/unified-basketball-analysis.git
cd unified-basketball-analysis
```

### Step 2: Clone All Sub-Projects

```bash
# Clone Shot Analysis
git clone https://github.com/hardik0/AI-basketball-analysis.git

# Clone Shot Detection
git clone https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker.git

# Clone Dribble YOLOv8
git clone https://github.com/siddhi-lipare/Basketball_Dribbles_Count_Using_YOLOv8.git

# Clone Dribble Gradio
git clone https://huggingface.co/spaces/gangulya111/basket-bll-dribble
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

**Activate it:**
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### Step 4: Install CUDA and cuDNN

**Windows:**
1. Download CUDA from https://developer.nvidia.com/cuda-downloads
2. Download cuDNN from https://developer.nvidia.com/cudnn
3. Follow installation instructions

**Linux:**
```bash
# Check CUDA availability
nvidia-smi

# Install CUDA (Ubuntu example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Step 5: Install Python Dependencies

```bash
# Install main requirements
pip install -r requirements.txt

# Install from each sub-project
pip install -r AI-basketball-analysis/requirements.txt
pip install -r AI-Basketball-Shot-Detection-Tracker/requirements.txt
pip install -r Basketball_Dribbles_Count_Using_YOLOv8/requirements.txt
pip install -r basket-bll-dribble/requirements.txt
```

### Step 6: Install OpenPose

**This is the most complex part. OpenPose requires special installation:**

```bash
# Clone OpenPose
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
cd openpose
git submodule update --init --recursive

# Build OpenPose (requires CMake)
mkdir build
cd build
cmake ..
make -j`nproc`
```

**For Python bindings:**
```bash
cd python
pip install .
```

### Step 7: Setup Environment Variables

Create `.env` file:
```bash
touch .env
```

Add your API keys:
```
SHOTTRACKER_API_KEY=your_shottracker_key_here
ROBOFLOW_API_KEY=your_roboflow_key_here
OPENPOSE_PATH=/path/to/openpose
CUDA_HOME=/usr/local/cuda
```

### Step 8: Create Required Directories

```bash
mkdir uploads
mkdir output
mkdir static
mkdir models
```

### Step 9: Download Model Weights

Each project needs its model weights:

```bash
# For AI-basketball-analysis
cd AI-basketball-analysis
# Follow their README to download Faster R-CNN weights

# For Basketball_Dribbles_Count_Using_YOLOv8
cd ../Basketball_Dribbles_Count_Using_YOLOv8
# Download YOLOv8 weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

cd ..
```

## 🎮 Running the Application

### Start the Unified System

```bash
python unified_app.py
```

The application will start on `http://localhost:5000`

### Run Individual Components

**Shot Analysis:**
```bash
cd AI-basketball-analysis
python app.py
```
Access at: `http://localhost:5000`

**Dribble Counter (Gradio):**
```bash
cd basket-bll-dribble
python app_gradio.py
```
Access at: `http://localhost:7860`

### Using the Web Interface

1. Open browser to `http://localhost:5000`
2. Upload a basketball video or image
3. Select analysis type:
   - **All Analysis**: Runs all 4 systems
   - **Shot Analysis**: OpenPose analysis only
   - **Shot Detection**: Detection tracking only
   - **Dribble (YOLO)**: YOLOv8 dribble counting
   - **Dribble (Gradio)**: Advanced dribble analysis
4. Click "Analyze Basketball Video"
5. Wait for processing (may take several minutes)
6. View combined results

## 📁 Project Structure

```
unified-basketball-analysis/
├── venv/                                    # Virtual environment
├── AI-basketball-analysis/                  # Shot analysis project
│   ├── app.py
│   ├── requirements.txt
│   └── models/
├── AI-Basketball-Shot-Detection-Tracker/    # Shot detection project
│   ├── main.py
│   └── requirements.txt
├── Basketball_Dribbles_Count_Using_YOLOv8/  # Dribble counter (YOLO)
│   ├── main.py
│   └── requirements.txt
├── basket-bll-dribble/                      # Dribble counter (Gradio)
│   ├── app_gradio.py
│   └── requirements.txt
├── templates/
│   └── index.html                          # Web interface
├── uploads/                                # Uploaded videos
├── output/                                 # Analysis results
├── static/                                 # Static assets
├── unified_app.py                          # Main application
├── requirements.txt                        # All dependencies
├── .env                                    # Environment variables
└── README.md                               # This file
```

## 🔌 API Integration

### ShotTracker API

The system integrates with ShotTracker's professional basketball analytics API.

**Endpoint:** `POST /shottracker_api`

**Example Request:**
```bash
curl -X POST http://localhost:5000/shottracker_api \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "shot",
    "player_id": "12345",
    "shot_result": "make",
    "shot_type": "3pt"
  }'
```

## 🔧 Troubleshooting

### CUDA Not Found

**Error:** `CUDA not available` or `No CUDA devices found`

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Check CUDA version
nvcc --version

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### OpenPose Not Working

**Error:** `Cannot import OpenPose`

**Solution:**
1. Ensure CUDA is properly installed
2. Rebuild OpenPose with Python bindings
3. Check OPENPOSE_PATH in .env file
4. Verify GPU drivers are up to date

### Module Not Found

**Error:** `ModuleNotFoundError: No module named 'xyz'`

**Solution:**
```bash
# Reinstall all requirements
pip install -r requirements.txt
pip install -r AI-basketball-analysis/requirements.txt
pip install -r AI-Basketball-Shot-Detection-Tracker/requirements.txt
pip install -r Basketball_Dribbles_Count_Using_YOLOv8/requirements.txt
pip install -r basket-bll-dribble/requirements.txt
```

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Change port in unified_app.py
app.run(port=5001)  # Use different port

# Or kill the process using the port (Linux/Mac)
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Out of Memory

**Error:** `CUDA out of memory`

**Solution:**
1. Reduce batch size in model configs
2. Process shorter video clips
3. Use CPU fallback (slower)
4. Upgrade GPU

### Model Weights Not Found

**Error:** `FileNotFoundError: Model weights not found`

**Solution:**
1. Download required model weights
2. Place them in correct directories
3. Update paths in config files

## 📝 License

This unified system combines multiple projects, each with their own licenses:
- AI Basketball Analysis: CMU OpenPose License (non-commercial research only)
- Other components: See individual project licenses

**Important:** OpenPose is for non-commercial research use only.

## 🤝 Contributing

Since this combines multiple existing projects, please contribute to the original repositories:
- https://github.com/hardik0/AI-basketball-analysis
- https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker
- https://github.com/siddhi-lipare/Basketball_Dribbles_Count_Using_YOLOv8
- https://huggingface.co/spaces/gangulya111/basket-bll-dribble

## 📧 Support

For issues related to:
- **Unified system**: Open an issue in this repository
- **Individual projects**: Contact original project maintainers
- **ShotTracker API**: Visit https://developer.shottracker.com

## 🎯 Next Steps

1. Test with sample videos
2. Configure model parameters
3. Customize web interface
4. Set up ShotTracker API integration
5. Add custom analysis features