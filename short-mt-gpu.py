import os
import sys
import time
import threading
import gc
import requests
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from funasr import AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
# 模型路径
MODEL_DIR = r"I:\AI\APP\FunASR\models\SenseVoiceSmall"
VAD_MODEL_DIR = "fsmn-vad"  # 使用 FunASR 内置 VAD 模型
# FFmpeg 路径
FFMPEG_PATH = r"I:\AI\APP\FunASR\ffmpeg\bin"
# LM Studio 默认地址
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
# GPU设置：是否使用GPU (True=使用GPU, False=使用CPU)
USE_GPU = True
# GPU设备号
GPU_DEVICE = "0"
# VAD 参数
VAD_MIN_SILENCE_DURATION = 0.3  # 静音持续时间阈值（秒），低于此值认为是句子间隔
VAD_MIN_SPEECH_DURATION = 0.1   # 最小语音片段持续时间（秒）
# ==========================================

# 将 FFmpeg 添加到环境变量
os.environ["PATH"] += os.pathsep + FFMPEG_PATH


class ShortVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("字幕生成工具 (VAD分段+SenseVoice GPU版)")
        self.root.geometry("920x980")
        self.asr_model = None  # SenseVoice 模型
        self.vad_model = None   # VAD 模型
        self.is_processing = False
        self.llm_lock = threading.Lock()
        self.setup_ui()
        self.check_lm_studio_connection()
        self.check_gpu_status()

    def setup_ui(self):
        # GPU设置区域
        frame_gpu = tk.LabelFrame(self.root, text="GPU设置", padx=10, pady=10)
        frame_gpu.pack(fill="x", padx=10, pady=5)

        self.var_use_gpu = tk.BooleanVar(value=USE_GPU)
        tk.Checkbutton(
            frame_gpu,
            text="启用GPU加速 (大幅提升转录速度)",
            variable=self.var_use_gpu,
            command=self.on_gpu_toggle
        ).grid(row=0, column=0, sticky="w", padx=5)

        tk.Label(frame_gpu, text="GPU设备号:").grid(row=0, column=1, padx=(20, 5))
        self.spin_gpu = tk.Spinbox(frame_gpu, from_=0, to=7, width=3)
        self.spin_gpu.delete(0, tk.END)
        self.spin_gpu.insert(0, GPU_DEVICE)
        self.spin_gpu.grid(row=0, column=2, sticky="w")

        self.lbl_gpu_status = tk.Label(frame_gpu, text="", fg="blue")
        self.lbl_gpu_status.grid(row=0, column=3, padx=10)

        # 输入选择
        frame_input = tk.LabelFrame(self.root, text="输入设置", padx=10, pady=10)
        frame_input.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_input, text="选择文件/文件夹:").grid(row=0, column=0, sticky="w")
        self.entry_input = tk.Entry(frame_input, width=50)
        self.entry_input.grid(row=0, column=1, padx=5)
        tk.Button(frame_input, text="选择文件", command=self.select_files).grid(row=0, column=2, padx=2)
        tk.Button(frame_input, text="选择文件夹", command=self.select_folder).grid(row=0, column=3, padx=2)

        # 输出设置
        frame_output = tk.LabelFrame(self.root, text="输出设置", padx=10, pady=10)
        frame_output.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_output, text="保存位置:").grid(row=0, column=0, sticky="w")
        self.entry_output = tk.Entry(frame_output, width=38)
        self.entry_output.grid(row=0, column=1, padx=5, sticky="w")
        tk.Button(frame_output, text="选择文件夹", command=self.select_output_folder).grid(row=0, column=2, padx=2)

        tk.Label(frame_output, text="字幕格式:").grid(row=0, column=3, padx=(15, 5), sticky="w")
        self.combo_format = ttk.Combobox(frame_output, width=8, state="readonly", values=["srt", "ass"])
        self.combo_format.set("srt")
        self.combo_format.grid(row=0, column=4, sticky="w", padx=5)

        tk.Label(frame_output, text="(文件名: 原视频名.srt)").grid(row=0, column=5, padx=5, sticky="w")

        # VAD 设置
        frame_vad = tk.LabelFrame(self.root, text="VAD 分段设置", padx=10, pady=10)
        frame_vad.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_vad, text="静音间隔阈值(秒):").grid(row=0, column=0, sticky="w")
        self.spin_vad_silence = tk.Spinbox(frame_vad, from_=0.1, to=2.0, increment=0.1, width=6)
        self.spin_vad_silence.delete(0, tk.END)
        self.spin_vad_silence.insert(0, "0.3")
        self.spin_vad_silence.grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame_vad, text="(句子间的静音时长)").grid(row=0, column=2, sticky="w")

        tk.Label(frame_vad, text="最小语音时长(秒):").grid(row=0, column=3, sticky="w", padx=(20, 5))
        self.spin_vad_speech = tk.Spinbox(frame_vad, from_=0.05, to=1.0, increment=0.05, width=6)
        self.spin_vad_speech.delete(0, tk.END)
        self.spin_vad_speech.insert(0, "0.1")
        self.spin_vad_speech.grid(row=0, column=4, sticky="w", padx=5)
        tk.Label(frame_vad, text="(过滤短噪音)").grid(row=0, column=5, sticky="w")

        # 处理设置
        frame_proc = tk.LabelFrame(self.root, text="处理设置", padx=10, pady=10)
        frame_proc.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_proc, text="并行线程数:").grid(row=0, column=0, sticky="w")
        self.spin_threads = tk.Spinbox(frame_proc, from_=1, to=8, width=5)
        self.spin_threads.delete(0, tk.END)
        self.spin_threads.insert(0, "2")
        self.spin_threads.grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame_proc, text="(GPU建议2-4)").grid(row=0, column=2, sticky="w")

        tk.Label(frame_proc, text="源视频语言:").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_lang = ttk.Combobox(frame_proc, width=10, state="readonly", values=["auto", "zh", "en", "yue", "ja", "ko"])
        self.combo_lang.set("auto")
        self.combo_lang.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # LM Studio 设置
        frame_lm = tk.LabelFrame(self.root, text="LM Studio 设置 (校对模型)", padx=10, pady=10)
        frame_lm.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_lm, text="IP地址:").grid(row=0, column=0, sticky="w")
        self.entry_lm_ip = tk.Entry(frame_lm, width=30)
        self.entry_lm_ip.insert(0, LM_STUDIO_URL)
        self.entry_lm_ip.grid(row=0, column=1, padx=5)
        tk.Button(frame_lm, text="检测连接", command=self.check_lm_studio_connection).grid(row=0, column=2, padx=5)
        self.lbl_status = tk.Label(frame_lm, text="状态: 未检测", fg="grey")
        self.lbl_status.grid(row=0, column=3, padx=5)
        tk.Label(frame_lm, text="选择模型:").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_models = ttk.Combobox(frame_lm, width=27, state="readonly")
        self.combo_models.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # 提示词设置
        frame_prompt = tk.LabelFrame(self.root, text="校对提示词 (字幕格式优化)", padx=10, pady=10)
        frame_prompt.pack(fill="x", padx=10, pady=5)
        self.text_prompt = scrolledtext.ScrolledText(frame_prompt, height=7, font=("Arial", 9))
        self.text_prompt.pack(fill="x")

        default_prompt = (
            "你是一个专业的字幕校对助手。请处理我提供的字幕文本，要求如下：\n"
            "1. 删除所有无意义的标记字符，例如 <|zh|>、<|HAPPY|>、<|BGM|>、<|withitn|> 等类似格式的无效内容。\n"
            "2. 保持字幕的简洁性，每条字幕不超过15个汉字或40个英文单词。\n"
            "3. 根据语义和逻辑适当断句，确保每条字幕语义完整。\n"
            "4. 移除或简化冗余的连接词和重复表达。\n"
            "5. 保留原句的核心含义和说话者的语气特点。\n"
            "6. 如果原文有明显错误，进行合理修正。\n"
            "7. 只输出处理后的字幕文本，每条字幕占一行，用【SPLIT】分隔不同字幕行。\n"
            "格式示例：\n"
            "第一行字幕内容【SPLIT】第二行字幕内容【SPLIT】第三行字幕内容\n"
            "{text}"
        )
        self.text_prompt.insert(tk.END, default_prompt)

        # 操作按钮
        frame_action = tk.Frame(self.root, padx=10, pady=5)
        frame_action.pack(fill="x")
        self.btn_start = tk.Button(frame_action, text="开始生成字幕", command=self.start_processing, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.btn_start.pack(fill="x", pady=5)

        # 日志区
        frame_log = tk.LabelFrame(self.root, text="详细日志", padx=10, pady=10)
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(frame_log, height=10, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("warn", foreground="orange")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("gpu", foreground="blue")
        self.log_text.tag_config("vad", foreground="purple")

    def log(self, message, level="info"):
        """带时间戳和颜色的日志输出"""
        def append_log():
            timestamp = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
            log_msg = f"[{timestamp}] {message}\n"
            self.log_text.insert(tk.END, log_msg, level)
            self.log_text.see(tk.END)
        self.root.after(0, append_log)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv")])
        if files:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, ";".join(files))
            if not self.entry_output.get():
                self.entry_output.insert(0, os.path.dirname(files[0]))

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, folder)
            if not self.entry_output.get():
                self.entry_output.insert(0, folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, folder)

    def check_gpu_status(self):
        """检查GPU状态"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                status_text = f"GPU可用: {gpu_name} | CUDA {cuda_version} | {vram:.1f}GB显存"
                self.lbl_gpu_status.config(text=status_text, fg="green")
                self.log(f"检测到GPU: {gpu_name}, CUDA {cuda_version}, {vram:.1f}GB显存", "gpu")
            else:
                self.lbl_gpu_status.config(text="CUDA不可用，将使用CPU", fg="orange")
                self.log("警告: CUDA不可用，将使用CPU模式", "warn")
        except ImportError:
            self.lbl_gpu_status.config(text="PyTorch未安装或CUDA支持缺失", fg="red")
            self.log("错误: PyTorch未安装或不支持CUDA", "error")
        except Exception as e:
            self.lbl_gpu_status.config(text=f"GPU检测失败", fg="red")
            self.log(f"GPU检测失败: {e}", "error")

    def on_gpu_toggle(self):
        """GPU开关切换时的处理"""
        if self.var_use_gpu.get():
            self.log("已启用GPU加速模式", "gpu")
            self.spin_threads.delete(0, tk.END)
            self.spin_threads.insert(0, "2")
        else:
            self.log("已切换到CPU模式", "info")
            self.spin_threads.delete(0, tk.END)
            self.spin_threads.insert(0, "4")

    def check_lm_studio_connection(self):
        ip = self.entry_lm_ip.get()
        self.log(f"正在检测 LM Studio 连接: {ip} ...")
        try:
            response = requests.get(f"{ip}/models", timeout=3)
            if response.status_code == 200:
                models = [m['id'] for m in response.json().get('data', [])]
                self.combo_models['values'] = models
                if models:
                    self.combo_models.current(0)
                self.lbl_status.config(text="状态: 已连接", fg="green")
                self.log(f"LM Studio 连接成功！发现模型: {len(models)} 个", "success")
            else:
                self.lbl_status.config(text="状态: 连接失败", fg="red")
                self.log("LM Studio 连接失败", "error")
        except Exception as e:
            self.lbl_status.config(text="状态: 连接错误", fg="red")
            self.log(f"连接错误: {e}", "error")

    def load_models(self):
        """加载 VAD 模型和 ASR 模型"""
        use_gpu = self.var_use_gpu.get()
        gpu_device = self.spin_gpu.get()
        device_str = f"cuda:{gpu_device}" if use_gpu else "cpu"
        mode_str = "GPU" if use_gpu else "CPU"

        # 加载 VAD 模型
        if self.vad_model is None:
            self.log(f"正在加载 VAD 模型 ({mode_str}模式)...", "vad")
            try:
                self.vad_model = AutoModel(
                    model=VAD_MODEL_DIR,
                    device=device_str,
                    disable_update=True,
                    ncpu=8 if not use_gpu else 1,
                )
                self.log("VAD 模型加载完成！", "success")
            except Exception as e:
                self.log(f"VAD 模型加载失败: {e}", "error")
                return False

        # 加载 SenseVoice 模型（不使用内置VAD）
        if self.asr_model is None:
            self.log(f"正在加载 SenseVoice 模型 ({mode_str}模式)...", "info")
            load_start = time.time()
            try:
                self.asr_model = AutoModel(
                    model=MODEL_DIR,
                    device=device_str,
                    disable_update=True,
                    vad_model=None,  # 不使用内置VAD，我们用单独的VAD模型
                    ncpu=8 if not use_gpu else 1,
                )
                load_end = time.time()
                self.log(f"SenseVoice 模型加载完成！耗时: {load_end - load_start:.2f}s", "success")

                if use_gpu:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mem_allocated = torch.cuda.memory_allocated(int(gpu_device)) / (1024**3)
                            mem_reserved = torch.cuda.memory_reserved(int(gpu_device)) / (1024**3)
                            self.log(f"GPU显存使用: {mem_allocated:.2f}GB / {mem_reserved:.2f}GB", "gpu")
                    except:
                        pass
            except Exception as e:
                error_msg = str(e)
                if "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
                    self.log(f"GPU加载失败，尝试CPU模式: {e}", "warn")
                    device_str = "cpu"
                    try:
                        self.vad_model = AutoModel(model=VAD_MODEL_DIR, device="cpu", disable_update=True)
                        self.asr_model = AutoModel(model=MODEL_DIR, device="cpu", disable_update=True, vad_model=None)
                        self.log("已切换到CPU模式", "success")
                    except Exception as e2:
                        self.log(f"CPU模式加载失败: {e2}", "error")
                        return False
                else:
                    self.log(f"模型加载失败: {e}", "error")
                    return False

        return True

    def detect_speech_segments(self, audio_path):
        """使用 VAD 模型检测语音段落"""
        try:
            import soundfile as sf

            # 读取音频文件
            waveform, sample_rate = sf.read(audio_path, dtype='float32')

            # 如果是立体声，取单通道
            if len(waveform.shape) > 1:
                waveform = waveform[:, 0]

            self.log(f"音频采样率: {sample_rate}Hz, 时长: {len(waveform)/sample_rate:.2f}秒", "vad")

            # 调用 VAD 模型进行语音检测
            res = self.vad_model.generate(
                input=audio_path,
                cache={},
                is_final=True,
            )

            segments = []
            if res and len(res) > 0:
                result = res[0] if isinstance(res, list) else res

                # 尝试从结果中提取时间戳
                if isinstance(result, dict):
                    # VAD 输出格式: [[beg1, end1], [beg2, end2], ...]
                    if 'timestamp' in result:
                        timestamp = result['timestamp']
                        if timestamp and isinstance(timestamp, list):
                            for item in timestamp:
                                if isinstance(item, (list, tuple)) and len(item) >= 2:
                                    beg = float(item[0]) if item[0] is not None else 0
                                    end = float(item[1]) if item[1] is not None else 0
                                    duration = end - beg
                                    # 过滤过短的片段
                                    if duration >= VAD_MIN_SPEECH_DURATION:
                                        segments.append({'start': beg, 'end': end})
                    elif isinstance(result, list):
                        for item in result:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                beg = float(item[0]) if item[0] is not None else 0
                                end = float(item[1]) if item[1] is not None else 0
                                duration = end - beg
                                if duration >= VAD_MIN_SPEECH_DURATION:
                                    segments.append({'start': beg, 'end': end})

            self.log(f"VAD 检测到 {len(segments)} 个语音段落", "vad")
            return segments

        except ImportError:
            # 如果没有 soundfile，使用 model.generate 直接处理
            self.log("使用 VAD 模型直接检测...", "vad")
            res = self.vad_model.generate(
                input=audio_path,
                cache={},
                is_final=True,
            )

            segments = []
            if res and len(res) > 0:
                result = res[0] if isinstance(res, list) else res
                if isinstance(result, dict) and 'timestamp' in result:
                    timestamp = result['timestamp']
                    if timestamp and isinstance(timestamp, list):
                        for item in timestamp:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                beg = float(item[0]) if item[0] is not None else 0
                                end = float(item[1]) if item[1] is not None else 0
                                if end - beg >= VAD_MIN_SPEECH_DURATION:
                                    segments.append({'start': beg, 'end': end})
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            beg = float(item[0]) if item[0] is not None else 0
                            end = float(item[1]) if item[1] is not None else 0
                            if end - beg >= VAD_MIN_SPEECH_DURATION:
                                segments.append({'start': beg, 'end': end})

            self.log(f"VAD 检测到 {len(segments)} 个语音段落", "vad")
            return segments

        except Exception as e:
            self.log(f"VAD 检测失败: {e}", "error")
            return []

    def recognize_segment(self, audio_path, start_time, end_time, lang):
        """对单个音频片段进行识别"""
        try:
            import soundfile as sf

            # 读取完整音频
            waveform, sample_rate = sf.read(audio_path, dtype='float32')

            # 如果是立体声，取单通道
            if len(waveform.shape) > 1:
                waveform = waveform[:, 0]

            # 计算采样点
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # 提取片段
            segment_waveform = waveform[start_sample:end_sample]

            # 保存临时文件
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            sf.write(temp_path, segment_waveform, sample_rate)

            # 使用 SenseVoice 识别
            language = lang if lang != "auto" else None
            res = self.asr_model.generate(
                input=temp_path,
                cache={},
                language=language,
                use_itn=True,
            )

            # 删除临时文件
            try:
                os.unlink(temp_path)
            except:
                pass

            # 提取识别结果
            text = ""
            if res and len(res) > 0:
                result = res[0] if isinstance(res, list) else res
                if isinstance(result, dict):
                    text = result.get('text', '')
                else:
                    text = str(result)

            return text.strip()

        except ImportError:
            # 如果没有 soundfile，直接传入时间范围（部分模型支持）
            language = lang if lang != "auto" else None
            try:
                # 尝试传入片段参数
                res = self.asr_model.generate(
                    input=audio_path,
                    cache={},
                    language=language,
                    use_itn=True,
                    segment_len=end_time - start_time,
                    segment_index=0,
                )

                text = ""
                if res and len(res) > 0:
                    result = res[0] if isinstance(res, list) else res
                    if isinstance(result, dict):
                        text = result.get('text', '')
                    else:
                        text = str(result)
                return text.strip()
            except:
                return ""
        except Exception as e:
            self.log(f"识别片段失败: {e}", "warn")
            return ""

    def correct_text_with_llm(self, text):
        """调用 LLM 进行校对"""
        ip = self.entry_lm_ip.get()
        model_id = self.combo_models.get()
        if not model_id:
            return text, "跳过 (未选模型)"

        prompt_template = self.text_prompt.get("1.0", tk.END).strip()
        if not prompt_template:
            return text, "跳过 (提示词为空)"

        if "{text}" in prompt_template:
            prompt = prompt_template.replace("{text}", text)
        else:
            prompt = prompt_template + f"\n\n{text}"

        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        acquired = self.llm_lock.acquire(timeout=300)
        if not acquired:
            return text, "LLM服务繁忙(超时)"

        try:
            t_start = time.time()
            response = requests.post(f"{ip}/chat/completions", headers=headers, json=data, timeout=120)
            t_end = time.time()
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                return result, f"成功 (耗时 {t_end - t_start:.1f}s)"
            else:
                return text, f"API错误 {response.status_code}"
        except Exception as e:
            return text, f"网络错误"
        finally:
            self.llm_lock.release()

    def generate_srt(self, segments, output_path):
        """生成SRT格式字幕文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = self.format_time_srt(seg['start'])
                end = self.format_time_srt(seg['end'])
                text = seg['text'].strip()
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")

    def generate_ass(self, segments, output_path, title="Subtitle"):
        """生成ASS格式字幕文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[Script Info]\n")
            f.write(f"Title: {title}\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write("PlayDepth: 0\n\n")

            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")

            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for seg in segments:
                start = self.format_time_ass(seg['start'])
                end = self.format_time_ass(seg['end'])
                text = seg['text'].strip().replace('\n', ' ')
                text = text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

    def format_time_srt(self, seconds):
        """将秒数转换为SRT时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def format_time_ass(self, seconds):
        """将秒数转换为ASS时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def process_single_file(self, video_path, idx, total):
        """单个文件的处理逻辑"""
        filename = os.path.basename(video_path)
        self.log(f"[线程] 开始处理 [{idx}/{total}]: {filename}")

        output_dir = self.entry_output.get()
        if not output_dir:
            output_dir = os.path.dirname(video_path)

        subtitle_format = self.combo_format.get()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base_name}.{subtitle_format}"
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(output_path):
            self.log(f"  > 跳过已存在: {output_name}", "warn")
            return

        try:
            task_start = time.time()
            lang = self.combo_lang.get()

            # ===== 步骤1: 使用 VAD 检测语音段落 =====
            self.log(f"  > [{filename}] 开始 VAD 语音检测...", "vad")
            vad_start = time.time()
            vad_segments = self.detect_speech_segments(video_path)
            vad_time = time.time() - vad_start
            self.log(f"  > [{filename}] VAD 检测完成，耗时: {vad_time:.2f}s", "vad")

            if not vad_segments:
                self.log(f"  > [{filename}] 警告: 未检测到语音段落", "warn")
                return

            # ===== 步骤2: 对每个段落进行识别 =====
            self.log(f"  > [{filename}] 开始逐段识别，共 {len(vad_segments)} 段...", "info")

            # 获取 VAD 参数
            try:
                self.vad_model = self.vad_model  # 确保已加载
            except:
                pass

            recognized_segments = []
            total_segments = len(vad_segments)

            for i, seg in enumerate(vad_segments):
                seg_start = seg['start']
                seg_end = seg['end']

                # 对每个片段进行识别
                text = self.recognize_segment(video_path, seg_start, seg_end, lang)

                if text:
                    recognized_segments.append({
                        'start': seg_start,
                        'end': seg_end,
                        'text': text
                    })

                # 每10段输出一次进度
                if (i + 1) % 10 == 0 or i + 1 == total_segments:
                    self.log(f"  > [{filename}] 识别进度: {i+1}/{total_segments} ({100*(i+1)//total_segments}%)", "info")

            asr_time = time.time() - task_start
            self.log(f"  > [{filename}] ASR完成，识别到 {len(recognized_segments)} 段，耗时: {asr_time:.2f}s", "success")

            if not recognized_segments:
                self.log(f"  > [{filename}] 警告: 未能识别到文字内容", "warn")
                return

            # ===== 步骤3: LLM 校对 =====
            # 合并所有文本进行校对
            raw_text = " ".join([seg['text'] for seg in recognized_segments])

            self.log(f"  > [{filename}] 等待 LLM 校对...")
            llm_start = time.time()
            corrected_text, status = self.correct_text_with_llm(raw_text)
            llm_time = time.time() - llm_start
            self.log(f"  > [{filename}] 校对结果: {status}", "info")

            # ===== 步骤4: 将校对后的文本分配回时间戳 =====
            corrected_lines = corrected_text.split('【SPLIT】')
            corrected_lines = [line.strip() for line in corrected_lines if line.strip()]

            if corrected_lines and recognized_segments:
                total_corrected = len(corrected_lines)
                total_orig = len(recognized_segments)

                if total_corrected >= total_orig:
                    # 使用前N个
                    for i in range(min(total_orig, total_corrected)):
                        recognized_segments[i]['text'] = corrected_lines[i]
                else:
                    # 合并分配
                    step = total_orig / total_corrected
                    for i, line in enumerate(corrected_lines):
                        start_idx = int(i * step)
                        end_idx = min(int((i + 1) * step), total_orig)
                        for j in range(start_idx, end_idx):
                            recognized_segments[j]['text'] = line if j == start_idx else ""
                        if start_idx < total_orig:
                            recognized_segments[start_idx]['text'] = line

            # 过滤空白片段
            final_segments = [seg for seg in recognized_segments if seg['text'].strip()]

            # ===== 步骤5: 保存字幕文件 =====
            if subtitle_format == 'srt':
                self.generate_srt(final_segments, output_path)
            else:
                self.generate_ass(final_segments, output_path, title=base_name)

            total_time = time.time() - task_start
            self.log(f"  > [{filename}] 字幕生成完成！总耗时: {total_time:.2f}s -> {output_name}", "success")

        except Exception as e:
            self.log(f"  > 处理出错 {filename}: {e}", "error")
            import traceback
            self.log(f"  > 详细错误: {traceback.format_exc()}", "error")
        finally:
            if self.var_use_gpu.get():
                try:
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass

    def start_processing(self):
        if self.is_processing:
            return
        inputs = self.entry_input.get()
        if not inputs:
            self.log("错误：请先选择输入文件或文件夹！", "error")
            return

        try:
            max_workers = int(self.spin_threads.get())
        except:
            max_workers = 2 if self.var_use_gpu.get() else 4

        self.is_processing = True
        self.btn_start.config(state="disabled", text="处理中...")
        threading.Thread(target=self.process_manager, args=(inputs, max_workers), daemon=True).start()

    def process_manager(self, inputs, max_workers):
        """管理线程池"""
        try:
            if not self.load_models():
                return

            files_to_process = []
            if os.path.isdir(inputs):
                for f in os.listdir(inputs):
                    if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')):
                        files_to_process.append(os.path.join(inputs, f))
            else:
                files_to_process = inputs.split(";")

            total = len(files_to_process)
            mode = "GPU" if self.var_use_gpu.get() else "CPU"
            format_info = self.combo_format.get().upper()
            self.log(f"======== 任务开始，共 {total} 个文件，并发数: {max_workers}，加速模式: {mode}，输出格式: {format_info} ========")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, video_path in enumerate(files_to_process, 1):
                    future = executor.submit(self.process_single_file, video_path, idx, total)
                    futures.append(future)
                for future in as_completed(futures):
                    pass

        except Exception as e:
            self.log(f"管理线程出错: {e}", "error")
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_start.config(state="normal", text="开始生成字幕"))
            self.log("\n======== 所有任务处理完毕 ========")


if __name__ == "__main__":
    root = tk.Tk()
    app = ShortVideoApp(root)
    root.mainloop()
