"""
Unified Basketball Analysis System - Working Version
Integrates the actual basketball analysis projects
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import subprocess
import json
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Import webcam detector
sys.path.append('.')
from webcam_yolo import generate_frames

# Ensure folders exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Check which projects are available
PROJECT_PATHS = {
    'shot_analysis': 'AI-basketball-analysis',
    'shot_detection': 'AI-Basketball-Shot-Detection-Tracker',
    'dribble_yolo': 'Basketball_Dribbles_Count_Using_YOLOv8',
    'dribble_gradio': 'basket-bll-dribble'
}

AVAILABLE_PROJECTS = {}
for name, path in PROJECT_PATHS.items():
    if os.path.exists(path):
        AVAILABLE_PROJECTS[name] = path
        print(f"✓ {name} is available at {path}")
    else:
        print(f"✗ {name} not found at {path}")


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html',
                         shot_analysis='shot_analysis' in AVAILABLE_PROJECTS,
                         shot_detection='shot_detection' in AVAILABLE_PROJECTS,
                         dribble_yolo='dribble_yolo' in AVAILABLE_PROJECTS,
                         dribble_gradio='dribble_gradio' in AVAILABLE_PROJECTS)


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Unified endpoint for video analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    analysis_type = request.form.get('analysis_type', 'all')
    
    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    results = {
        'filename': file.filename,
        'filepath': filepath,
        'analyses': {}
    }
    
    # Run Shot Analysis (AI-basketball-analysis)
    if analysis_type in ['all', 'shot_analysis'] and 'shot_analysis' in AVAILABLE_PROJECTS:
        try:
            result = run_shot_analysis(filepath)
            results['analyses']['shot_analysis'] = result
        except Exception as e:
            results['analyses']['shot_analysis'] = {'error': str(e)}
    
    # Run Shot Detection (AI-Basketball-Shot-Detection-Tracker)
    if analysis_type in ['all', 'shot_detection'] and 'shot_detection' in AVAILABLE_PROJECTS:
        try:
            result = run_shot_detection(filepath)
            results['analyses']['shot_detection'] = result
        except Exception as e:
            results['analyses']['shot_detection'] = {'error': str(e)}
    
    # Run Dribble Counter (YOLOv8)
    if analysis_type in ['all', 'dribble_yolo'] and 'dribble_yolo' in AVAILABLE_PROJECTS:
        try:
            result = run_dribble_yolo(filepath)
            results['analyses']['dribble_yolo'] = result
        except Exception as e:
            results['analyses']['dribble_yolo'] = {'error': str(e)}
    
    # Run Dribble Counter (Gradio)
    if analysis_type in ['all', 'dribble_gradio'] and 'dribble_gradio' in AVAILABLE_PROJECTS:
        try:
            result = run_dribble_gradio(filepath)
            results['analyses']['dribble_gradio'] = result
        except Exception as e:
            results['analyses']['dribble_gradio'] = {'error': str(e)}
    
    return jsonify(results)


@app.route('/webcam')
def webcam_page():
    """Real-time webcam analysis page"""
    return render_template('webcam.html',
                         dribble_yolo='dribble_yolo' in AVAILABLE_PROJECTS,
                         shot_detection='shot_detection' in AVAILABLE_PROJECTS)


@app.route('/video_feed')
def video_feed():
    """Video streaming route using YOLOv8"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_shot_analysis(filepath):
    """
    Run AI-basketball-analysis
    This project has app.py as its main file
    """
    project_path = AVAILABLE_PROJECTS['shot_analysis']
    
    # For now, return info that it's available but needs to be run separately
    return {
        'status': 'available',
        'message': 'Shot analysis with OpenPose requires GPU and separate execution',
        'instructions': f'To run: cd {project_path} && python app.py',
        'note': 'This will start a separate Flask server for shot analysis'
    }


def run_shot_detection(filepath):
    """
    Run AI-Basketball-Shot-Detection-Tracker
    Uses main.py for detection
    """
    project_path = AVAILABLE_PROJECTS['shot_detection']
    abs_filepath = os.path.abspath(filepath)
    
    # Change to project directory and run
    original_dir = os.getcwd()
    try:
        os.chdir(project_path)
        
        # Run the detection script
        result = subprocess.run(
            ['python', 'main.py', '--video', abs_filepath],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        os.chdir(original_dir)
        
        return {
            'status': 'completed' if result.returncode == 0 else 'error',
            'stdout': result.stdout,
            'stderr': result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        return {'status': 'error', 'message': 'Analysis timed out after 5 minutes'}
    except Exception as e:
        os.chdir(original_dir)
        return {'status': 'error', 'message': str(e)}


def run_dribble_yolo(filepath):
    """
    Run Basketball_Dribbles_Count_Using_YOLOv8
    Uses the notebook's logic converted to script
    """
    project_path = AVAILABLE_PROJECTS['dribble_yolo']
    
    return {
        'status': 'available',
        'message': 'Dribble counting with YOLOv8',
        'instructions': f'This project uses a Jupyter notebook. See {project_path}/dribble-count.ipynb',
        'note': 'Can be converted to a Python script if needed'
    }


def run_dribble_gradio(filepath):
    """
    Run basket-bll-dribble
    Uses app_gradio.py
    """
    project_path = AVAILABLE_PROJECTS['dribble_gradio']
    
    return {
        'status': 'available',
        'message': 'Dribble counter with Gradio interface',
        'instructions': f'To run: cd {project_path} && python app_gradio.py',
        'note': 'This will start a Gradio interface on port 7860',
        'url': 'http://localhost:7860'
    }


@app.route('/run_project/<project_name>')
def run_project(project_name):
    """Launch individual projects"""
    if project_name not in AVAILABLE_PROJECTS:
        return jsonify({'error': 'Project not found'}), 404
    
    project_path = AVAILABLE_PROJECTS[project_name]
    
    instructions = {
        'shot_analysis': f'cd {project_path} && python app.py',
        'shot_detection': f'cd {project_path} && python main.py --video <your_video>',
        'dribble_yolo': f'cd {project_path} && jupyter notebook dribble-count.ipynb',
        'dribble_gradio': f'cd {project_path} && python app_gradio.py'
    }
    
    return jsonify({
        'project': project_name,
        'path': project_path,
        'instruction': instructions.get(project_name, 'No instructions available')
    })


@app.route('/outputs/<path:filename>')
def download_output(filename):
    """Serve output files"""
    # Handle both direct filenames and paths with folders
    if '/' in filename:
        filename = filename.split('/')[-1]
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/status')
def status():
    """Check status of all projects"""
    status_info = {}
    
    for name, path in AVAILABLE_PROJECTS.items():
        status_info[name] = {
            'available': True,
            'path': path,
            'files': os.listdir(path)[:5]  # First 5 files
        }
    
    return jsonify(status_info)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏀 Unified Basketball Analysis System")
    print("="*60)
    print("\nAvailable Projects:")
    for name, path in AVAILABLE_PROJECTS.items():
        print(f"  ✓ {name}: {path}")
    
    print("\n" + "="*60)
    print("Main Interface: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
import os
port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)