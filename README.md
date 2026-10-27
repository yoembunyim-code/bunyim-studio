# BUNYIM Dubbing Studio (Desktop)

កម្មវិធី Windows សម្រាប់បកប្រែវីដេអូ និងបញ្ចូលសំឡេងខ្មែរ (AI Dubbing)។

## ដំណើរការ
1. ចាប់សំឡេងដោយ **Whisper លើកុំព្យូទ័រអ្នក** (គ្មាន API, គ្មាន 503; លើកដំបូងទាញម៉ូដែល)
2. បកប្រែជាខ្មែរ ដោយ Gemini (បើមាន key) ឬ Google Translate (ឥតគិតថ្លៃ)
3. កែអត្ថបទ/ពេលវេលា/តួអង្គ ក្នុងតារាង និង timeline
4. បង្កើតសំឡេងខ្មែរ (Edge TTS: ពិសិដ្ឋ, ស្រីមុំ) ហើយ Export MP4

## ដំឡើងពី GitHub (ងាយបំផុត)
1. ដាក់ឯកសារទាំងអស់នេះក្នុង GitHub repo
2. Actions → "Build BUNYIM Windows EXE" → Run workflow
3. រង់ចាំ ~10-15 នាទី រួចទាញ **BunyimStudio-windows** ពី Artifacts → Unzip → ចុច `BunyimStudio.exe`

## ដំណើរការពី source (Windows)
```
pip install -r requirements.txt
python main.py
```
ត្រូវមាន ffmpeg ក្នុង PATH (ផ្នែក .exe មាន ffmpeg ភ្ជាប់ស្រាប់)។

## សាកល្បងកូដ (គ្មាន GUI)
```
python tests/test_engine.py
python tests/smoke_gui.py
```
