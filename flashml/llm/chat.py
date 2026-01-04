def detect_model_type(path):
    """
    Detect if a model requires Unsloth.
    
    Returns:
        True if unsloth is needed, False otherwise
    """
    from pathlib import Path
    
    path_str = str(path)
    
    # Check if "unsloth" is in the path
    if "unsloth" in path_str.lower():
        return True
    
    # Check if it's an adapter with unsloth base model
    adapter_config_path = Path(path) / "adapter_config.json"
    if adapter_config_path.exists():
        import json
        try:
            with open(adapter_config_path) as f:
                config = json.load(f)
                base_model = config.get("base_model_name_or_path", "")
                if "unsloth" in base_model.lower():
                    return True
        except Exception:
            pass
    
    return False


def create_chat_window_class():
    """Create the ChatWindow class with proper PyQt6 inheritance"""
    from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QFont, QTextCursor
    
    class ChatWindow(QMainWindow):
        def __init__(self, model, tokenizer, system_prompt, temperature, top_p, top_k, engine):
            super().__init__()
            
            self.model = model
            self.tokenizer = tokenizer
            self.system_prompt = system_prompt
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.engine = engine
            
            self.conversation_history = []
            self.is_generating = False
            self.current_response = ""
            self.token_times = []
            self.generation_start = 0
            
            # Create signals object
            class Signals(QObject):
                token_generated = pyqtSignal(str)
                generation_finished = pyqtSignal()
                error_occurred = pyqtSignal(str)
                speed_update = pyqtSignal(float)
            
            self.signals = Signals()
            self.signals.token_generated.connect(self.append_token)
            self.signals.generation_finished.connect(self.generation_complete)
            self.signals.error_occurred.connect(self.show_error)
            self.signals.speed_update.connect(self.update_speed_display)
            
            self.init_ui()
            
        def init_ui(self):
            """Initialize the modern UI"""
            self.setWindowTitle("AI Chat Interface")
            self.setGeometry(100, 100, 900, 700)
            
            # Apply modern dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1e1e1e;
                }
                QTextEdit {
                    background-color: #2d2d2d;
                    color: #e0e0e0;
                    border: none;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                }
                QLineEdit {
                    background-color: #2d2d2d;
                    color: #e0e0e0;
                    border: 2px solid #3d3d3d;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                }
                QLineEdit:focus {
                    border: 2px solid #5e5ce6;
                }
                QPushButton {
                    background-color: #5e5ce6;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #6e6cf6;
                }
                QPushButton:pressed {
                    background-color: #4e4cd6;
                }
                QPushButton:disabled {
                    background-color: #3d3d3d;
                    color: #666;
                }
                QLabel {
                    color: #e0e0e0;
                    font-size: 13px;
                }
            """)
            
            # Central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            main_layout.setSpacing(12)
            main_layout.setContentsMargins(20, 20, 20, 20)
            
            # Top bar with speedometer
            top_bar = QHBoxLayout()
            
            title_label = QLabel("[AI Chat]")
            title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #5e5ce6;")
            top_bar.addWidget(title_label)
            
            top_bar.addStretch()
            
            self.speed_label = QLabel("Speed: 0.0 tok/s")
            self.speed_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4ade80;")
            top_bar.addWidget(self.speed_label)
            
            engine_label = QLabel(f"Engine: {self.engine}")
            engine_label.setStyleSheet("font-size: 12px; color: #888;")
            top_bar.addWidget(engine_label)
            
            main_layout.addLayout(top_bar)
            
            # Chat display area
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            self.chat_display.setFont(QFont("Segoe UI", 11))
            main_layout.addWidget(self.chat_display)
            
            # Input area
            input_layout = QHBoxLayout()
            input_layout.setSpacing(12)
            
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Type your message here...")
            self.input_field.returnPressed.connect(self.send_message)
            input_layout.addWidget(self.input_field)
            
            self.send_button = QPushButton("Send")
            self.send_button.setFixedWidth(100)
            self.send_button.clicked.connect(self.send_message)
            input_layout.addWidget(self.send_button)
            
            main_layout.addLayout(input_layout)
            
            # Initial message
            self.add_system_message("Welcome! Start chatting with the AI model.")
            
            # Display system prompt if provided
            if self.system_prompt:
                self.add_system_message(f"[System Prompt] {self.system_prompt}")
            
        def add_system_message(self, text):
            """Add a system message to the chat"""
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertHtml(
                f'<div style="color: #888; font-style: italic; margin: 10px 0;">{text}</div>'
            )
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
            
        def add_user_message(self, text):
            """Add a user message to the chat"""
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertHtml(
                f'<div style="background-color: #5e5ce6; color: white; padding: 12px; '
                f'border-radius: 12px; margin: 10px 0; margin-left: 20%;">'
                f'<b>You:</b><br>{text}</div>'
            )
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
            
        def start_assistant_message(self):
            """Start a new assistant message bubble"""
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertHtml(
                f'<div style="background-color: #3d3d3d; color: #e0e0e0; padding: 12px; '
                f'border-radius: 12px; margin: 10px 0; margin-right: 20%;">'
                f'<b>Assistant:</b><br>'
            )
            
        def append_token(self, token):
            """Append a token to the current response"""
            self.current_response += token
            
            # Insert the token at the end
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertPlainText(token)
            
            # Scroll to bottom
            self.chat_display.verticalScrollBar().setValue(
                self.chat_display.verticalScrollBar().maximum()
            )
            
        def generation_complete(self):
            """Handle generation completion"""
            self.is_generating = False
            self.send_button.setEnabled(True)
            self.input_field.setEnabled(True)
            self.input_field.setFocus()
            
            # Save to history
            self.conversation_history.append({
                "role": "assistant",
                "content": self.current_response
            })
            
            # Close the div
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.insertHtml('</div>')
            
            self.current_response = ""
            self.token_times = []
            
        def show_error(self, error_msg):
            """Show an error message"""
            self.is_generating = False
            self.send_button.setEnabled(True)
            self.input_field.setEnabled(True)
            self.add_system_message(f"[ERROR] {error_msg}")
            
        def update_speed_display(self, speed):
            """Update the token speed display"""
            self.speed_label.setText(f"Speed: {speed:.1f} tok/s")
            
        def send_message(self):
            """Send a message and get a response"""
            from threading import Thread
            
            if self.is_generating:
                return
                
            user_message = self.input_field.text().strip()
            if not user_message:
                return
                
            # Add user message to display
            self.add_user_message(user_message)
            
            # Add to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Clear input
            self.input_field.clear()
            self.input_field.setEnabled(False)
            self.send_button.setEnabled(False)
            
            # Start assistant response
            self.is_generating = True
            self.current_response = ""
            self.start_assistant_message()
            
            # Start generation in a separate thread
            thread = Thread(target=self.generate_response, daemon=True)
            thread.start()
            
        def generate_response(self):
            """Generate response in a separate thread"""
            import time
            from threading import Thread
            
            try:
                # Build messages
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.extend(self.conversation_history)
                
                # Format for the model
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    # Fallback formatting
                    prompt = ""
                    for msg in messages:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "system":
                            prompt += f"System: {content}\n\n"
                        elif role == "user":
                            prompt += f"User: {content}\n\n"
                        elif role == "assistant":
                            prompt += f"Assistant: {content}\n\n"
                    prompt += "Assistant: "
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Generate with streaming
                self.generation_start = time.time()
                token_count = 0
                
                if self.engine == "transformers":
                    from transformers import TextIteratorStreamer
                    
                    streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                    
                    generation_kwargs = dict(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=512,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        do_sample=True if self.temperature > 0 else False,
                    )
                    
                    # Start generation in another thread
                    gen_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                    gen_thread.start()
                    
                    # Stream tokens
                    for token in streamer:
                        if token:
                            self.signals.token_generated.emit(token)
                            token_count += 1
                            
                            # Calculate speed
                            elapsed = time.time() - self.generation_start
                            if elapsed > 0:
                                speed = token_count / elapsed
                                self.signals.speed_update.emit(speed)
                    
                    gen_thread.join()
                    
                else:  # unsloth
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        do_sample=True if self.temperature > 0 else False,
                    )
                    
                    # Decode and emit token by token for visualization
                    response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    words = response.split()
                    
                    for i, word in enumerate(words):
                        token = word + (" " if i < len(words) - 1 else "")
                        self.signals.token_generated.emit(token)
                        token_count += 1
                        
                        # Calculate speed
                        elapsed = time.time() - self.generation_start
                        if elapsed > 0:
                            speed = token_count / elapsed
                            self.signals.speed_update.emit(speed)
                        
                        time.sleep(0.05)  # Simulate streaming for visual effect
                
                self.signals.generation_finished.emit()
                
            except Exception as e:
                self.signals.error_occurred.emit(str(e))
    
    return ChatWindow


def chat(
    path,
    system_prompt="",
    temperature=0.7,
    top_p=0.95,
    top_k=200,
):
    """
    Launch a modern chat interface for interacting with a language model.
    
    Args:
        path: Path to model checkpoint (local or HuggingFace) or adapter
        system_prompt: System prompt for the conversation
        temperature: Sampling temperature (0.0 to 2.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    """
    import sys
    
    path_str = str(path)
    
    # Detect model type
    is_unsloth = detect_model_type(path)
    
    print(f"Loading model from: {path}")
    print(f"Engine detected: {'Unsloth' if is_unsloth else 'Transformers'}")
    
    # Import and load model based on type
    if is_unsloth:
        # Import unsloth FIRST before anything else
        try:
            from unsloth import FastLanguageModel
            UNSLOTH_AVAILABLE = True
        except Exception as e:
            UNSLOTH_AVAILABLE = False
            print("⚠️  Unsloth import failed. Falling back to Transformers.")
            print(f"Error: {e}")
            print("\nTo fix unsloth, try:")
            print("  pip uninstall -y zstd")
            print("  pip install --upgrade --force-reinstall zstd")
            print("  pip install --upgrade unsloth\n")
            is_unsloth = False
        
        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=path_str,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            engine = "unsloth"
    
    if not is_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(path_str)
        model = AutoModelForCausalLM.from_pretrained(
            path_str,
            device_map="auto",
            torch_dtype="auto",
        )
        model.eval()
        engine = "transformers"
    
    print("Model loaded successfully!")
    
    # Launch Qt application
    from PyQt6.QtWidgets import QApplication
    
    # Create the ChatWindow class with proper PyQt6 inheritance
    ChatWindow = create_chat_window_class()
    
    app = QApplication(sys.argv)
    window = ChatWindow(model, tokenizer, system_prompt, temperature, top_p, top_k, engine)
    window.show()
    sys.exit(app.exec())