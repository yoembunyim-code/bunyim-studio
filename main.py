"""Movie Time Studio (Desktop) - ចំណុចចាប់ផ្តើម"""
import engine

engine.setup_ffmpeg_path()

if __name__ == "__main__":
    from gui import run

    run()
