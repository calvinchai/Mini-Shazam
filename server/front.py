from flask import Flask, request, jsonify,render_template
import os
from flask_cors import CORS

from pydub import AudioSegment


app = Flask(__name__,
            static_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

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
        
        
        return jsonify(cough_detected=cough_detected), 200
    else:
        return jsonify(error="No audio file provided"), 400


@app.route('/<path:path>')
def static_file(path):
    print(path)
    return app.send_static_file(path)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
