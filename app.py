import os  #модуль, позволяющий работать напрямую с ос, управлять папками и тд
import sys #
import time #
import shutil #
import subprocess #
from flask import Flask, request, jsonify, render_template, send_from_directory #
from gtts import gTTS #
import imageio_ffmpeg #

app = Flask(__name__)

# 1. Настраиваем рабочий FFmpeg автоматически, чтобы SadTalker и pydub не падали
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
project_root = os.path.abspath(".")
local_ffmpeg = os.path.join(project_root, "ffmpeg.exe")
local_ffprobe = os.path.join(project_root, "ffprobe.exe")

# Копируем рабочий бинарник в корень проекта, если его там еще нет
if not os.path.exists(local_ffmpeg):
    shutil.copyfile(ffmpeg_exe, local_ffmpeg)
if not os.path.exists(local_ffprobe):
    shutil.copyfile(ffmpeg_exe, local_ffprobe)

# Добавляем корень проекта в системный PATH, чтобы подпроцессы SadTalker видели ffmpeg.exe
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# Папки проекта
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
AI_MODEL_DIR = 'SadTalker'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs("temp", exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def generate_lip_sync_video(image_path, audio_path, output_path):
    """
    Запуск нейросети SadTalker и перекодировка видео для браузера (H.264).
    """
    abs_image_path = os.path.abspath(image_path)
    abs_audio_path = os.path.abspath(audio_path)
    abs_model_dir = os.path.abspath(AI_MODEL_DIR)

    if not os.path.isdir(abs_model_dir):
        raise Exception(f"Папка нейросети не найдена по пути: {abs_model_dir}")

    # Создаем отдельную временную папку под этот конкретный запуск,
    # чтобы не перепутать новое видео со старыми файлами
    run_temp_dir = os.path.abspath(os.path.join("temp", f"run_{int(time.time())}"))
    os.makedirs(run_temp_dir, exist_ok=True)

    cmd = [
        sys.executable, #выдает путь файла
        "inference.py",
        "--driven_audio", abs_audio_path,
        "--source_image", abs_image_path,
        "--result_dir", run_temp_dir,
        "--still",
        "--preprocess", "full"
    ]

    try:
        # Передаем обновленное окружение с рабочим PATH в подпроцесс
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=abs_model_dir,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Лог нейросети SadTalker:", result.stdout)

        # Ищем сгенерированный .mp4 файл во временной папке запуска
        found_videos = []
        for root, dirs, files in os.walk(run_temp_dir):
            for file in files:
                if file.endswith('.mp4'):
                    found_videos.append(os.path.join(root, file))

        if not found_videos:
            raise Exception("Видеофайл не был создан нейросетью SadTalker.")

        # Берем самый последний (финальный) mp4 файл
        raw_video = max(found_videos, key=os.path.getmtime)

        # ПЕРЕКОДИРОВКА ДЛЯ БРАУЗЕРА:
        # Конвертируем в кодек H.264 (libx264) + yuv420p и принудительно прикрепляем чистый звук,
        # чтобы в Chrome / Edge / Safari гарантированно отображалась картинка, а не только аудиоплеер
        abs_output_path = os.path.abspath(output_path)
        ffmpeg_cmd = [
            local_ffmpeg, "-y",
            "-i", raw_video,
            "-i", abs_audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            abs_output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Очищаем временную папку запуска
        shutil.rmtree(run_temp_dir, ignore_errors=True)

        if not os.path.exists(abs_output_path):
            raise Exception("Ошибка при финальной сборке видео через FFmpeg.")

    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения: {e.stderr}")
        raise Exception(f"Ошибка ML-модели: {e.stderr}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files or 'text' not in request.form:
        return jsonify({'error': 'Необходимо загрузить изображение и ввести текст'}), 400

    image_file = request.files['image']
    text_content = request.form['text']

    if image_file.filename == '' or not text_content.strip():
        return jsonify({'error': 'Пустой текст или файл не выбран'}), 400

    # Сохраняем входное фото
    image_filename = "input_face.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_file.save(image_path)

    # Генерируем озвучку в MP3 и сразу переводим в чистый WAV (чтобы pydub читал его без ffprobe)
    timestamp = int(time.time())
    mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{timestamp}.mp3")
    audio_filename = f"voice_{timestamp}.wav"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

    tts = gTTS(text=text_content, lang='ru')
    tts.save(mp3_path)

    # Конвертируем mp3 -> wav локальным ffmpeg
    subprocess.run(
        [local_ffmpeg, "-y", "-i", mp3_path, audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if os.path.exists(mp3_path):
        os.remove(mp3_path)

    # Путь для финального видео
    output_filename = f"result_{timestamp}.mp4"
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

    try:
        generate_lip_sync_video(image_path, audio_path, output_path)
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500

    return jsonify({
        'status': 'success',
        'video_url': f'/results/{output_filename}',
        'audio_url': f'/uploads/{audio_filename}'
    })


@app.route('/results/<filename>')
def get_result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, mimetype='video/mp4')


@app.route('/uploads/<filename>')
def get_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)