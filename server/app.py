from flask import Flask, request, jsonify
import os
from flask_cors import CORS

from pydub import AudioSegment
from model.model import predict_audio

app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route('/status', methods=['GET'])
def status():
    return jsonify(status="online"), 200

@app.route('/audio', methods=['POST'])
def audio():
    audio_file = request.files['audio']
    if audio_file:
        # Save the audio file
        audio_path = os.path.join("uploads", audio_file.filename+".webm")
        audio_file.save(audio_path)
        print(audio_file.read())
        #convert to wav
        sound = AudioSegment.from_file(audio_path, format="webm")
        sound.export(audio_path.replace(".webm", ".wav"), format="wav")
        
        cough_detected = predict_audio(audio_path.replace(".webm", ".wav")) == 1
        
        return jsonify(cough_detected=cough_detected), 200
    else:
        return jsonify(error="No audio file provided"), 400

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000)
