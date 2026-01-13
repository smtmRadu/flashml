import sys
import os
import time
import torch
from threading import Thread
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLineEdit, QPushButton, QLabel, 
                             QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Global cache
_MODEL_CACHE = {}

# -------------------------------------------------------------------------
#  Worker Thread
# -------------------------------------------------------------------------

class GenerationWorker(QThread):
    new_token_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    metrics_signal = pyqtSignal(float)

    def __init__(self, model_path, user_input, history, temperature, top_p, top_k, system_prompt):
        super().__init__()
        self.model_path = model_path
        self.user_input = user_input
        self.history = history
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt

    def run(self):
        global _MODEL_CACHE
        
        try:
            start_time = time.time()
            token_count = 0
            
            # Load Model (Cached)
            if self.model_path not in _MODEL_CACHE:
                print(f"Loading model: {self.model_path}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                # CRITICAL FIX: Set pad_token_id to prevent CUDA errors
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    pad_token_id=tokenizer.pad_token_id  # Pass to model config
                )
                
                if device == "cuda":
                    model = model.to("cuda")
                
                _MODEL_CACHE[self.model_path] = {"model": model, "tokenizer": tokenizer}
                print(f"Model loaded. Vocab size: {len(tokenizer)}, Pad ID: {tokenizer.pad_token_id}")
            
            cached = _MODEL_CACHE[self.model_path]
            model = cached["model"]
            tokenizer = cached["tokenizer"]
            
            # Build message list for chat template
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            for sender, msg in self.history:
                role = "user" if sender == "User" else "assistant"
                messages.append({"role": role, "content": msg})
            
            messages.append({"role": "user", "content": self.user_input})
            
            # Use model's native chat template (Gemma 3 compatible)
            conversation_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize with attention mask
            inputs = tokenizer(
                conversation_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Print debug info
            print(f"Input shape: {inputs['input_ids'].shape}, Max token ID: {inputs['input_ids'].max().item()}")
            
            # Setup streaming
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                streamer=streamer,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,  # Prevents degenerate loops
            )
            
            # Generate in separate thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
            thread.start()
            
            # Stream tokens
            for new_text in streamer:
                token_count += 1
                self.new_token_signal.emit(new_text)
            
            end_time = time.time()
            duration = end_time - start_time
            speed = token_count / duration if duration > 0 else 0
            self.metrics_signal.emit(speed)

        except Exception as e:
            self.new_token_signal.emit(f"\n[Error: {str(e)}]")
        finally:
            self.finished_signal.emit()

# -------------------------------------------------------------------------
#  Frontend
# -------------------------------------------------------------------------

class ChatBubble(QFrame):
    def __init__(self, text, is_user=False):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Segoe UI", 10))
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        self.layout.addWidget(self.label)
        self.layout.setContentsMargins(15, 10, 15, 10)

        if is_user:
            color = "#0078D4"
            text_col = "white"
        else:
            color = "#2D2D2D"
            text_col = "#E0E0E0"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 15px;
                border-{('bottom-right' if is_user else 'bottom-left')}-radius: 0px;
            }}
            QLabel {{
                color: {text_col};
                background-color: transparent;
            }}
        """)

class MainWindow(QMainWindow):
    def __init__(self, model_path="gpt2", system_prompt="You are a helpful AI assistant.", 
                 temperature=0.7, top_p=0.9, top_k=50):
        super().__init__()
        
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.history = []

        self.init_ui()
        self.setup_styles()
        QTimer.singleShot(100, self.preload_model)

    def init_ui(self):
        self.setWindowTitle("Neural Chat")
        self.resize(500, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        header_layout = QHBoxLayout()
        self.model_label = QLabel(f"Model: {self.model_path}")
        self.model_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.model_label.setStyleSheet("color: #FFFFFF;")
        
        self.speed_label = QLabel("Speed: - t/s")
        self.speed_label.setStyleSheet("color: #888888; font-size: 11px;")
        
        header_layout.addWidget(self.model_label)
        header_layout.addStretch()
        header_layout.addWidget(self.speed_label)
        main_layout.addLayout(header_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()
        
        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area)

        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #252526; border-radius: 20px;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 5, 10, 5)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setStyleSheet("border: none; color: white; background: transparent; font-size: 14px;")
        self.input_field.returnPressed.connect(self.send_message)

        self.send_btn = QPushButton("Send")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedSize(60, 30)
        self.send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        main_layout.addWidget(input_frame)

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QScrollArea { background-color: #1E1E1E; border: none; }
            QWidget { background-color: #1E1E1E; }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #198CDD; }
            QPushButton:disabled { background-color: #444444; color: #888888; }
        """)

    def preload_model(self):
        if self.model_path not in _MODEL_CACHE:
            self.model_label.setText(f"Loading {self.model_path}...")
            
            def load():
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    
                    if tokenizer.pad_token_id is None:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    
                    torch_dtype = torch.float16 if device == "cuda" else torch.float32
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch_dtype,
                        device_map=None,
                        low_cpu_mem_usage=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    if device == "cuda":
                        model = model.to("cuda")
                    
                    _MODEL_CACHE[self.model_path] = {"model": model, "tokenizer": tokenizer}
                    QTimer.singleShot(0, lambda: self.model_label.setText(f"Model: {self.model_path}"))
                except Exception as e:
                    print(f"Model loading error: {e}")
                    QTimer.singleShot(0, lambda: self.model_label.setText(f"Error loading model"))
            
            Thread(target=load, daemon=True).start()

    def add_message(self, text, is_user):
        bubble = ChatBubble(text, is_user)
        h_layout = QHBoxLayout()
        if is_user:
            h_layout.addStretch()
            h_layout.addWidget(bubble)
        else:
            h_layout.addWidget(bubble)
            h_layout.addStretch()
        
        self.chat_layout.addLayout(h_layout)
        QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))
        return bubble

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.add_message(text, is_user=True)
        self.input_field.clear()
        self.input_field.setDisabled(True)
        self.send_btn.setDisabled(True)
        self.speed_label.setText("Generating...")

        self.current_ai_bubble = self.add_message("", is_user=False)
        self.current_ai_text = ""

        self.worker = GenerationWorker(
            model_path=self.model_path,
            user_input=text,
            history=self.history,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            system_prompt=self.system_prompt
        )
        self.worker.new_token_signal.connect(self.update_ai_message)
        self.worker.metrics_signal.connect(self.update_metrics)
        self.worker.finished_signal.connect(self.generation_finished)
        self.worker.start()

        self.history.append(("User", text))

    def update_ai_message(self, token):
        self.current_ai_text += token
        self.current_ai_bubble.label.setText(self.current_ai_text)
        scrollbar = self.scroll_area.verticalScrollBar()
        if scrollbar.value() >= scrollbar.maximum() - 20:
            scrollbar.setValue(scrollbar.maximum())

    def update_metrics(self, speed):
        self.speed_label.setText(f"Speed: {speed:.2f} tok/s")

    def generation_finished(self):
        self.history.append(("Assistant", self.current_ai_text))
        self.input_field.setDisabled(False)
        self.send_btn.setDisabled(False)
        self.input_field.setFocus()

# -------------------------------------------------------------------------
#  Main Entry Point
# -------------------------------------------------------------------------

def chat(model_path: str = "gpt2", system_prompt: str = "You are a helpful AI assistant.",
         temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50):
    """
    Launch the Neural Chat application.
    
    Args:
        model_path: HuggingFace model ID or local path
        system_prompt: System instruction for the AI
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter (>0)
    """
    app = QApplication(sys.argv)
    window = MainWindow(
        model_path=model_path, 
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    chat()