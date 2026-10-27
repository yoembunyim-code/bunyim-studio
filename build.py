import PyInstaller.main

PyInstaller.main.run([
    '--noconfirm',
    '--clean',
    '--windowed',
    '--name', 'BunyimStudio',
    '--add-binary', 'bin/ffmpeg.exe;.',
    '--add-binary', 'bin/ffprobe.exe;.',
    '--collect-all', 'faster_whisper',
    '--collect-all', 'ctranslate2',
    '--collect-all', 'av',
    '--collect-all', 'edge_tts',
    '--collect-all', 'deep_translator',
    '--collect-all', 'google.genai',
    'main.py',
])
