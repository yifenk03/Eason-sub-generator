import os
import sys
import time
import threading
import gc  # <--- 新增：引入垃圾回收模块
import requests
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from funasr import AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
# 模型路径
MODEL_DIR = r"I:\AI\APP\FunASR\models\SenseVoiceSmall"
# FFmpeg 路径
FFMPEG_PATH = r"I:\AI\APP\FunASR\ffmpeg\bin"
# LM Studio 默认地址
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
# GPU设置：是否使用GPU (True=使用GPU, False=使用CPU)
USE_GPU = True
# GPU设备号 (如果有多个GPU可以修改，0表示第一块GPU)
GPU_DEVICE = "0"
# ==========================================

# 将 FFmpeg 添加到环境变量
os.environ["PATH"] += os.pathsep + FFMPEG_PATH


class ShortVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("字幕生成工具 (GPU加速版)")
        self.root.geometry("900x950")
        self.model = None
        self.is_processing = False
        # 线程锁，用于防止 LM Studio 并发冲突
        self.llm_lock = threading.Lock()
        self.setup_ui()
        self.check_lm_studio_connection()
        self.check_gpu_status()

    def setup_ui(self):
        # GPU设置区域 (新增)
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

        # 输出设置 (修改：添加字幕格式选择，与保存位置同一行)
        frame_output = tk.LabelFrame(self.root, text="输出设置", padx=10, pady=10)
        frame_output.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_output, text="保存位置:").grid(row=0, column=0, sticky="w")
        self.entry_output = tk.Entry(frame_output, width=40)
        self.entry_output.grid(row=0, column=1, padx=5, sticky="w")
        tk.Button(frame_output, text="选择文件夹", command=self.select_output_folder).grid(row=0, column=2, padx=2)

        tk.Label(frame_output, text="字幕格式:").grid(row=0, column=3, padx=(20, 5), sticky="w")
        self.combo_format = ttk.Combobox(frame_output, width=8, state="readonly", values=["srt", "ass"])
        self.combo_format.set("srt")
        self.combo_format.grid(row=0, column=4, sticky="w", padx=5)

        tk.Label(frame_output, text="(文件名: 原视频名.srt/ass)").grid(row=0, column=5, padx=5, sticky="w")

        # 处理设置
        frame_proc = tk.LabelFrame(self.root, text="处理设置", padx=10, pady=10)
        frame_proc.pack(fill="x", padx=10, pady=5)

        # 线程数
        tk.Label(frame_proc, text="并行线程数:").grid(row=0, column=0, sticky="w")
        self.spin_threads = tk.Spinbox(frame_proc, from_=1, to=8, width=5)
        self.spin_threads.delete(0, tk.END)
        self.spin_threads.insert(0, "2")  # GPU模式下建议降低线程数
        self.spin_threads.grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame_proc, text="(GPU模式下建议2-4)").grid(row=0, column=2, sticky="w")

        # 语言选择
        tk.Label(frame_proc, text="源视频语言:").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_lang = ttk.Combobox(frame_proc, width=10, state="readonly", values=["auto", "zh", "en", "yue", "ja", "ko"])
        self.combo_lang.set("auto")
        self.combo_lang.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # 批处理大小 (GPU优化参数)
        tk.Label(frame_proc, text="批处理大小(秒):").grid(row=1, column=2, sticky="w", padx=(20, 5))
        self.spin_batch = tk.Spinbox(frame_proc, from_=30, to=300, width=5)
        self.spin_batch.delete(0, tk.END)
        self.spin_batch.insert(0, "60")
        self.spin_batch.grid(row=1, column=3, sticky="w")
        tk.Label(frame_proc, text="(GPU建议60-120)").grid(row=1, column=4, sticky="w")

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

        # 提示词设置 (优化为字幕格式)
        frame_prompt = tk.LabelFrame(self.root, text="校对提示词 (字幕格式优化)", padx=10, pady=10)
        frame_prompt.pack(fill="x", padx=10, pady=5)
        self.text_prompt = scrolledtext.ScrolledText(frame_prompt, height=8, font=("Arial", 9))
        self.text_prompt.pack(fill="x")

        # 插入默认字幕校对提示词
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
        self.log_text = scrolledtext.ScrolledText(frame_log, height=12, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

        # 定义日志标签颜色
        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("warn", foreground="orange")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("gpu", foreground="blue")

    def log(self, message, level="info"):
        """带时间戳和颜色的日志输出 (线程安全)"""
        def append_log():
            timestamp = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
            log_msg = f"[{timestamp}] {message}\n"
            self.log_text.insert(tk.END, log_msg, level)
            self.log_text.see(tk.END)
        # 使用 after 确保 GUI 更新在主线程执行
        self.root.after(0, append_log)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv")])
        if files:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, ";".join(files))
            # 新增：自动设置默认输出路径为第一个文件所在目录
            if not self.entry_output.get():
                self.entry_output.insert(0, os.path.dirname(files[0]))

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, folder)
            # 新增：自动设置默认输出路径
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

                status_text = f"GPU可用: {gpu_name} | CUDA {cuda_version} | {vram:.1f}GB显存 | {gpu_count}块GPU"
                self.lbl_gpu_status.config(text=status_text, fg="green")
                self.log(f"检测到GPU: {gpu_name}, CUDA {cuda_version}, {vram:.1f}GB显存", "gpu")
            else:
                self.lbl_gpu_status.config(text="CUDA不可用，将使用CPU", fg="orange")
                self.log("警告: CUDA不可用，将使用CPU模式", "warn")
        except ImportError:
            self.lbl_gpu_status.config(text="PyTorch未安装或CUDA支持缺失", fg="red")
            self.log("错误: PyTorch未安装或不支持CUDA", "error")
        except Exception as e:
            self.lbl_gpu_status.config(text=f"GPU检测失败: {str(e)[:30]}", fg="red")
            self.log(f"GPU检测失败: {e}", "error")

    def on_gpu_toggle(self):
        """GPU开关切换时的处理"""
        if self.var_use_gpu.get():
            self.log("已启用GPU加速模式", "gpu")
            self.spin_threads.delete(0, tk.END)
            self.spin_threads.insert(0, "2")  # GPU模式建议降低线程数
            self.spin_batch.delete(0, tk.END)
            self.spin_batch.insert(0, "60")
        else:
            self.log("已切换到CPU模式", "info")
            self.spin_threads.delete(0, tk.END)
            self.spin_threads.insert(0, "4")  # CPU模式可以提高线程数

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
                self.log("LM Studio 连接失败，请检查服务是否开启。", "error")
        except Exception as e:
            self.lbl_status.config(text="状态: 连接错误", fg="red")
            self.log(f"连接错误: {e}", "error")

    def load_asr_model(self):
        """加载 FunASR 模型，支持CPU/GPU自动切换"""
        if self.model is None:
            use_gpu = self.var_use_gpu.get()
            gpu_device = self.spin_gpu.get()

            if use_gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.log("警告: GPU不可用，尝试使用CPU模式...", "warn")
                        use_gpu = False
                except ImportError:
                    self.log("警告: PyTorch CUDA不可用，尝试使用CPU模式...", "warn")
                    use_gpu = False

            device_str = f"cuda:{gpu_device}" if use_gpu else "cpu"
            mode_str = "GPU" if use_gpu else "CPU"

            self.log(f"正在加载 FunASR 模型 ({mode_str}模式) 设备: {device_str}，请稍候...", "info")
            load_start = time.time()

            try:
                self.model = AutoModel(
                    model=MODEL_DIR,
                    device=device_str,
                    disable_update=True,
                    vad_model="fsmn-vad",  # <--- 修改为启用 VAD 模型
                    # punc_model="ct-punc", # 建议同时也加上标点模型，效果更好
                    ncpu=8 if not use_gpu else 1,
                )

                load_end = time.time()
                self.log(f"模型加载完成！耗时: {load_end - load_start:.2f}s", "success")

                if use_gpu:
                    self.log(f"GPU模式已启用 - 使用 {device_str} 进行推理", "gpu")
                    # 显示GPU内存使用情况
                    try:
                        import torch
                        if torch.cuda.is_available():
                            mem_allocated = torch.cuda.memory_allocated(int(gpu_device)) / (1024**3)
                            mem_reserved = torch.cuda.memory_reserved(int(gpu_device)) / (1024**3)
                            self.log(f"GPU显存使用: 已分配 {mem_allocated:.2f}GB / 预留 {mem_reserved:.2f}GB", "gpu")
                    except:
                        pass
                else:
                    self.log("使用CPU模式进行推理", "info")

            except Exception as e:
                error_msg = str(e)
                if "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
                    self.log(f"GPU加载失败，尝试使用CPU模式: {e}", "warn")
                    try:
                        self.model = AutoModel(
                            model=MODEL_DIR,
                            device="cpu",
                            disable_update=True,
                            vad_model=None,
                        )
                        self.log("已成功切换到CPU模式", "success")
                        load_end = time.time()
                        self.log(f"模型加载完成！耗时: {load_end - load_start:.2f}s", "success")
                    except Exception as e2:
                        self.log(f"CPU模式也加载失败: {e2}", "error")
                        return False
                else:
                    self.log(f"模型加载失败: {e}", "error")
                    return False

        return True

    def correct_text_with_llm(self, text):
        """调用 LLM 进行校对 (带线程锁，防止 LM Studio 并发崩溃)"""
        ip = self.entry_lm_ip.get()
        model_id = self.combo_models.get()
        if not model_id:
            return text, "跳过 (未选模型)"
        # 从 GUI 获取提示词
        prompt_template = self.text_prompt.get("1.0", tk.END).strip()
        if not prompt_template:
            return text, "跳过 (提示词为空)"
        # 简单的占位符替换，兼容用户是否在提示词中写了 {text}
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
        # === 关键：加锁 ===
        # 多个线程可能同时完成 ASR，同时请求 LM Studio，这会导致本地服务阻塞或崩溃
        # 使用 Lock 确保同一时刻只有一个线程在调用 LLM API
        acquired = self.llm_lock.acquire(timeout=300)  # 设置5分钟超时防死锁
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
            return text, f"网络错误: {str(e)[:20]}..."
        finally:
            self.llm_lock.release()  # 释放锁

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
            # ASS文件头
            f.write("[Script Info]\n")
            f.write(f"Title: {title}\n")
            f.write("ScriptType: v4.00+\n")
            f.write("Collisions: Normal\n")
            f.write("PlayDepth: 0\n\n")

            # 字幕样式
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")

            # 字幕事件
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            for seg in segments:
                start = self.format_time_ass(seg['start'])
                end = self.format_time_ass(seg['end'])
                text = seg['text'].strip().replace('\n', ' ')
                # 转义ASS特殊字符
                text = text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

    def format_time_srt(self, seconds):
        """将秒数转换为SRT时间格式 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def format_time_ass(self, seconds):
        """将秒数转换为ASS时间格式 (HH:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def process_single_file(self, video_path, idx, total):
        """单个文件的处理逻辑，用于多线程执行"""
        filename = os.path.basename(video_path)
        self.log(f"[线程] 开始处理 [{idx}/{total}]: {filename}")
        # 确定输出路径
        output_dir = self.entry_output.get()
        if not output_dir:
            output_dir = os.path.dirname(video_path)

        # 获取字幕格式
        subtitle_format = self.combo_format.get()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base_name}.{subtitle_format}"
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(output_path):
            self.log(f"  > 跳过已存在: {output_name}", "warn")
            return
        try:
            task_start = time.time()
            # 1. ASR 转录
            # 获取语言设置
            lang = self.combo_lang.get()
            if lang == "auto":
                lang = None  # FunASR auto detection
            # 获取批处理大小
            try:
                batch_size = int(self.spin_batch.get())
            except:
                batch_size = 60

            res = self.model.generate(
                input=video_path,
                cache={},
                language=lang,
                use_itn=True,
                batch_size_s=batch_size  # GPU模式下使用较大的批处理大小
            )
            asr_time = time.time() - task_start
            self.log(f"  > [{filename}] ASR完成，耗时: {asr_time:.2f}s", "success")

            # 提取带时间戳的片段
            segments = []
            raw_text = ""

            if res and len(res) > 0:
                result_item = res[0] if isinstance(res, list) else res

                # 尝试提取时间戳信息
                if 'timestamp' in result_item:
                    timestamp = result_item['timestamp']
                    if timestamp and isinstance(timestamp, list):
                        for item in timestamp:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                start = item[0] if isinstance(item[0], (int, float)) else 0
                                end = item[1] if isinstance(item[1], (int, float)) else 0
                                text = item[2] if len(item) > 2 else ""
                                if text:
                                    segments.append({
                                        'start': float(start) if start else 0,
                                        'end': float(end) if end else 0,
                                        'text': str(text)
                                    })
                                    raw_text += str(text) + " "

                # 如果没有时间戳，尝试使用sentence级别的时间戳
                if not segments and 'sentence_info' in result_item:
                    sentence_info = result_item['sentence_info']
                    if sentence_info and isinstance(sentence_info, list):
                        for item in sentence_info:
                            if isinstance(item, dict):
                                start = item.get('start', 0)
                                end = item.get('end', 0)
                                text = item.get('text', '')
                                if text:
                                    segments.append({
                                        'start': float(start),
                                        'end': float(end),
                                        'text': str(text)
                                    })
                                    raw_text += str(text) + " "

                # 如果仍然没有时间戳，使用段落级别的时间戳
                if not segments and 'paragraphs' in result_item:
                    paragraphs = result_item['paragraphs']
                    if paragraphs and isinstance(paragraphs, list):
                        for para in paragraphs:
                            if isinstance(para, dict):
                                start = para.get('start', 0)
                                end = para.get('end', 0)
                                text = para.get('text', '')
                                if text:
                                    segments.append({
                                        'start': float(start),
                                        'end': float(end),
                                        'text': str(text)
                                    })
                                    raw_text += str(text) + " "

                # 最后尝试从text直接获取
                if not raw_text:
                    raw_text = result_item.get('text', '')

            if not raw_text.strip():
                self.log(f"  > [{filename}] 警告: 未识别到文字内容。", "warn")
                return

            # 2. LLM 校对 (内部已加锁)
            self.log(f"  > [{filename}] 等待 LLM 校对...")
            llm_start = time.time()
            corrected_text, status = self.correct_text_with_llm(raw_text)
            llm_time = time.time() - llm_start
            self.log(f"  > [{filename}] 校对结果: {status}", "info")

            # 3. 处理校对后的文本，按【SPLIT】分割
            corrected_lines = corrected_text.split('【SPLIT】')
            corrected_lines = [line.strip() for line in corrected_lines if line.strip()]

            # 4. 将校对后的文本重新分配到时间戳
            if segments and corrected_lines:
                # 按比例分配校对后的文本到原始时间戳
                total_segments = len(segments)
                total_corrected = len(corrected_lines)

                if total_corrected >= total_segments:
                    # 校对行数多于或等于原片段，使用前N个
                    for i in range(min(total_segments, total_corrected)):
                        segments[i]['text'] = corrected_lines[i]
                else:
                    # 校对行数少于原片段，平均分配
                    step = total_segments / total_corrected
                    for i, line in enumerate(corrected_lines):
                        start_idx = int(i * step)
                        end_idx = int((i + 1) * step) if i < total_corrected - 1 else total_segments
                        # 合并中间的时间段
                        combined_text = ' '.join([segments[j]['text'] for j in range(start_idx, end_idx)])
                        for j in range(start_idx, end_idx):
                            segments[j]['text'] = line if j == start_idx else ""
                        if start_idx < len(segments):
                            segments[start_idx]['text'] = line
            elif corrected_lines and not segments:
                # 没有时间戳信息，创建均等分配的时间段
                self.log(f"  > [{filename}] 警告: 无时间戳信息，使用均等分配", "warn")
                total_lines = len(corrected_lines)
                duration_estimate = asr_time / max(total_lines, 1)  # 估算每行时长
                segments = []
                for i, line in enumerate(corrected_lines):
                    start = i * duration_estimate
                    end = (i + 1) * duration_estimate
                    segments.append({
                        'start': start,
                        'end': end,
                        'text': line
                    })

            # 过滤空白片段
            segments = [seg for seg in segments if seg['text'].strip()]

            # 5. 保存字幕文件
            if subtitle_format == 'srt':
                self.generate_srt(segments, output_path)
            else:  # ass
                self.generate_ass(segments, output_path, title=base_name)

            total_time = time.time() - task_start
            self.log(f"  > [{filename}] 字幕生成完成！总耗时: {total_time:.2f}s -> {output_name}", "success")
        except Exception as e:
            self.log(f"  > 处理出错 {filename}: {e}", "error")
            import traceback
            self.log(f"  > 详细错误: {traceback.format_exc()}", "error")
        finally:
            # === 新增：自动清理 GPU 缓存 ===
            if self.var_use_gpu.get():
                try:
                    import torch
                    # 1. 先回收 Python 对象
                    gc.collect()
                    # 2. 清空 PyTorch CUDA 缓存
                    torch.cuda.empty_cache()
                    self.log(f"  > [{filename}] GPU缓存清理完成", "gpu")
                except Exception as e:
                    self.log(f"  > 清理GPU缓存时出错: {e}", "warn")

    def start_processing(self):
        if self.is_processing:
            return
        inputs = self.entry_input.get()
        if not inputs:
            self.log("错误：请先选择输入文件或文件夹！", "error")
            return
        # 获取线程数
        try:
            max_workers = int(self.spin_threads.get())
        except:
            max_workers = 2 if self.var_use_gpu.get() else 4

        self.is_processing = True
        self.btn_start.config(state="disabled", text="处理中...")
        # 启动后台线程来管理线程池，避免阻塞 GUI
        threading.Thread(target=self.process_manager, args=(inputs, max_workers), daemon=True).start()

    def process_manager(self, inputs, max_workers):
        """管理线程池"""
        try:
            if not self.load_asr_model():
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
            # 使用 ThreadPoolExecutor 并行处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx, video_path in enumerate(files_to_process, 1):
                    # 提交任务到线程池
                    future = executor.submit(self.process_single_file, video_path, idx, total)
                    futures.append(future)
                # 等待所有任务完成
                for future in as_completed(futures):
                    # 这里可以捕获异常，但 process_single_file 内部已经有 try-catch
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
