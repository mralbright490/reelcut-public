import os
import sys
import math
import threading
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ffmpeg
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Environment Setup & Warning Suppressions
# ---------------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Helper function to convert seconds to formatted HH:MM:SS string
def format_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# ---------------------------------------------------------------------------
# Core Processing Engine
# ---------------------------------------------------------------------------
class ReelCutEngine:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model = WhisperModel(model_size_or_path=model_size, device=device, compute_type="int8")

    def extract_audio(self, video_path: str, output_audio_path: str):
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_audio_path, ac=1, ar="16000", format="wav")
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

    def transcribe_audio(self, audio_path: str, status_callback=None):
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        segment_list = []
        for segment in segments:
            segment_list.append(segment)
            if status_callback:
                status_callback(f"Transcribed: [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text[:40]}...")
        return segment_list, info

    def cut_clip(self, input_video: str, start_time: float, duration: float, output_video: str):
        stream = ffmpeg.input(input_video, ss=start_time, t=duration)
        stream = ffmpeg.output(stream, output_video, c="copy")
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

    def export_transcript_txt(self, segments, output_txt_path: Path, source_name: str):
        """Generates a structured transcript file with timestamps and raw text."""
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(f"REELCUT TRANSCRIPT FILE: {source_name}\n")
            f.write("=" * 65 + "\n\n")
            f.write("--- TIMED DIALOGUE CUES ---\n\n")
            
            raw_paragraphs = []
            for seg in segments:
                start_str = format_time(seg.start)
                end_str = format_time(seg.end)
                text_clean = seg.text.strip()
                f.write(f"[{start_str} -> {end_str}]  {text_clean}\n")
                raw_paragraphs.append(text_clean)

            f.write("\n\n" + "=" * 65 + "\n")
            f.write("--- RAW SCRIPT TEXT (CONTINUOUS) ---\n")
            f.write("=" * 65 + "\n\n")
            f.write(" ".join(raw_paragraphs) + "\n")

# ---------------------------------------------------------------------------
# Graphical User Interface (GUI)
# ---------------------------------------------------------------------------
class ReelCutGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ReelCut Engine - Local AI Video Clipper (AP DreamStudios)")
        self.root.geometry("680x560")
        self.root.resizable(False, False)

        # Apply dark theme aesthetic
        self.root.configure(bg="#1e1e1e")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", foreground="#ffffff", background="#1e1e1e", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Horizontal.TProgressbar", background="#007acc", troughcolor="#333333")

        # File Selection Frame
        frame_file = tk.Frame(root, bg="#1e1e1e")
        frame_file.pack(fill="x", padx=15, pady=10)
        
        ttk.Label(frame_file, text="Target Video File:").pack(anchor="w")
        self.entry_path = tk.Entry(frame_file, bg="#2d2d2d", fg="#ffffff", insertbackground="white", font=("Segoe UI", 10))
        self.entry_path.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)
        ttk.Button(frame_file, text="Browse...", command=self.browse_file).pack(side="right")

        # Configuration Settings Frame
        frame_opts = tk.Frame(root, bg="#1e1e1e")
        frame_opts.pack(fill="x", padx=15, pady=5)

        ttk.Label(frame_opts, text="Clip Duration (Seconds):").grid(row=0, column=0, sticky="w", pady=5)
        self.slider_duration = tk.Scale(frame_opts, from_=15, to=120, orient="horizontal", bg="#1e1e1e", fg="#ffffff", highlightthickness=0)
        self.slider_duration.set(45)
        self.slider_duration.grid(row=0, column=1, sticky="ew", padx=10)

        # Script Export Checkbox
        self.var_export_txt = tk.BooleanVar(value=True)
        self.chk_export_txt = tk.Checkbutton(
            frame_opts, 
            text="Export Script (.txt)", 
            variable=self.var_export_txt, 
            bg="#1e1e1e", 
            fg="#ffffff", 
            selectcolor="#2d2d2d", 
            activebackground="#1e1e1e", 
            activeforeground="#ffffff",
            font=("Segoe UI", 10)
        )
        self.chk_export_txt.grid(row=0, column=2, padx=15, sticky="w")

        # Progress & Status Indicators
        frame_status = tk.Frame(root, bg="#1e1e1e")
        frame_status.pack(fill="x", padx=15, pady=10)

        self.progress = ttk.Progressbar(frame_status, mode="indeterminate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=5)

        self.lbl_status = ttk.Label(frame_status, text="Ready to process.", font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(anchor="w")

        # Log Console Display
        ttk.Label(root, text="Execution Logs:").pack(anchor="w", padx=15)
        self.txt_logs = tk.Text(root, height=12, bg="#121212", fg="#00ffcc", font=("Consolas", 9), insertbackground="white")
        self.txt_logs.pack(fill="both", padx=15, pady=5)

        # Action Button
        self.btn_run = tk.Button(root, text="Generate ReelCut Shorts & Script", bg="#007acc", fg="#ffffff", font=("Segoe UI", 11, "bold"), activebackground="#005999", activeforeground="#ffffff", command=self.start_processing_thread)
        self.btn_run.pack(fill="x", padx=15, pady=10)

    def log(self, message: str):
        self.txt_logs.insert(tk.END, f"{message}\n")
        self.txt_logs.see(tk.END)
        self.lbl_status.config(text=message)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov")])
        if filename:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, filename)

    def start_processing_thread(self):
        video_file = self.entry_path.get().strip()
        if not video_file or not Path(video_file).exists():
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        self.btn_run.config(state="disabled")
        self.progress.start(10)
        threading.Thread(target=self.run_pipeline, args=(video_file,), daemon=True).start()

    def run_pipeline(self, video_file: str):
        reelcut_dir = Path(__file__).parent.resolve()
        output_dir = reelcut_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = Path(video_file)
        temp_audio = reelcut_dir / f"{video_path.stem}_temp.wav"
        clip_duration = float(self.slider_duration.get())

        try:
            self.log("[1/4] Initializing AI Whisper Engine...")
            engine = ReelCutEngine()

            self.log("[2/4] Extracting temporary audio stream...")
            engine.extract_audio(str(video_path), str(temp_audio))

            self.log("[3/4] Transcribing audio and mapping clip intervals...")
            segments, info = engine.transcribe_audio(str(temp_audio), status_callback=self.log)

            if not segments:
                self.log("WARNING: No speech segments detected. Unable to extract clips.")
                return

            # Export transcript if option is checked
            if self.var_export_txt.get():
                txt_output_path = output_dir / f"{video_path.stem}_transcript.txt"
                self.log(f"--> Exporting script transcript to '{txt_output_path.name}'...")
                engine.export_transcript_txt(segments, txt_output_path, video_path.name)

            # Determine clip count and timeline checkpoints
            total_duration = info.duration if info.duration > 0 else segments[-1].end
            target_clip_count = 10 if total_duration >= 1800 else max(1, math.floor(total_duration / 180))

            self.log(f"[INFO] Video Duration: {total_duration/60:.1f} mins | Target Clips: {target_clip_count}")

            step_interval = total_duration / target_clip_count
            start_points = []

            for i in range(target_clip_count):
                target_time = i * step_interval
                best_segment = min(segments, key=lambda seg: abs(seg.start - target_time))
                if best_segment.start not in start_points:
                    start_points.append(best_segment.start)

            # Generate clips into output folder
            self.log(f"[4/4] Cutting {len(start_points)} short clips into 'output/' folder...")
            generated_clips = []

            for idx, start_sec in enumerate(start_points, start=1):
                out_name = f"{video_path.stem}_short_{idx:02d}.mp4"
                out_path = output_dir / out_name
                
                self.log(f"-> Generating Clip {idx}/{len(start_points)} at {start_sec:.1f}s...")
                engine.cut_clip(str(video_path), start_time=start_sec, duration=clip_duration, output_video=str(out_path))
                generated_clips.append(out_name)

            self.log(f"SUCCESS: Processing complete! Output saved to '{output_dir}'.")
            messagebox.showinfo("Success", f"Processing complete!\nClips and script exported to:\n{output_dir}")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Execution Error", str(e))
        finally:
            if temp_audio.exists():
                try:
                    temp_audio.unlink()
                    self.log("Cleaned up temporary audio files.")
                except Exception as clean_err:
                    self.log(f"Cleanup warning: {clean_err}")
            
            self.progress.stop()
            self.btn_run.config(state="normal")

# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Running in CLI Mode...")
    else:
        root = tk.Tk()
        app = ReelCutGUI(root)
        root.mainloop()