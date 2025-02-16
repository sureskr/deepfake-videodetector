from flask import Flask, render_template, request, jsonify
import os
from simple_video_detector import SimpleVideoDetector
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detector
detector = SimpleVideoDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'})
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video selected'})
        
    if video:
        try:
            filename = secure_filename(video.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(filepath)
            
            # Analyze video
            result = detector.detect_deepfake(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Error processing video: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 