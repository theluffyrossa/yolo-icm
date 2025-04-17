# Detector YOLO e Gerador de Resumos ICM Integrados
# Sistema que combina detecção de objetos (YOLO) com geração de resumos de imagens (ICM)
# Requisitos: pip install opencv-python numpy tkinter pillow torch transformers

import cv2
import os
import time
import numpy as np
import subprocess
import sys
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading

# Adicionando dependências para o ICM
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Constantes
YOUTUBE_URL = "https://www.youtube.com/watch?v=FO5ym9LPYYc"
SAVE_DIR = "deteccoes"  # Pasta onde serão salvos os arquivos (alterado para português)
MODEL_DIR = "yolo_model"
ICM_MODEL_DIR = "icm_model"

# Diretório para traduções em português
TRANSLATIONS_DIR = os.path.join(SAVE_DIR, "traducoes")

# URLs dos arquivos do YOLO
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
CLASSES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

# Variáveis globais
running = False
cap = None
net = None
output_layers = None
classes = None
icm_processor = None
icm_model = None

def download_yolo_files():
    """Baixar arquivos necessários para o YOLO"""
    config_path = os.path.join(MODEL_DIR, "yolov4.cfg")
    weights_path = os.path.join(MODEL_DIR, "yolov4.weights")
    classes_path = os.path.join(MODEL_DIR, "coco.names")
    
    # Criar diretórios
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(ICM_MODEL_DIR):
        os.makedirs(ICM_MODEL_DIR)
    if not os.path.exists(TRANSLATIONS_DIR):
        os.makedirs(TRANSLATIONS_DIR)
    
    # Baixar arquivo de configuração
    if not os.path.exists(config_path):
        print("Baixando arquivo de configuração YOLO...")
        try:
            urllib.request.urlretrieve(CONFIG_URL, config_path)
            print("Download do arquivo de configuração concluído!")
        except Exception as e:
            print(f"Erro ao baixar arquivo de configuração: {e}")
            return False, None, None, None
    
    # Baixar arquivo de classes
    if not os.path.exists(classes_path):
        print("Baixando arquivo de classes COCO...")
        try:
            urllib.request.urlretrieve(CLASSES_URL, classes_path)
            print("Download do arquivo de classes concluído!")
        except Exception as e:
            print(f"Erro ao baixar arquivo de classes: {e}")
            return False, None, None, None
    
    # Verificar arquivo de pesos (ou instruções para download)
    if not os.path.exists(weights_path):
        print("\nO arquivo de pesos YOLOv4 (~245MB) precisa ser baixado.")
        print("Baixando arquivo de pesos YOLOv4 (pode demorar)...")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
            print("Download do arquivo de pesos concluído!")
        except Exception as e:
            print(f"Erro ao baixar arquivo de pesos: {e}")
            print("Por favor, baixe manualmente de:")
            print(WEIGHTS_URL)
            print(f"E salve como: {weights_path}")
            return False, None, None, None
    
    # Carregar nomes das classes
    with open(classes_path, 'r') as f:
        classes_list = f.read().strip().split('\n')
    
    return True, config_path, weights_path, classes_list

def initialize_icm_model():
    """Inicializar o modelo de geração de resumos (ICM)"""
    global icm_processor, icm_model
    try:
        print("Carregando modelo ICM (Gerador de Resumos)...")
        # Carregar o processador e o modelo
        icm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=ICM_MODEL_DIR)
        icm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=ICM_MODEL_DIR)
        print("Modelo ICM carregado com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao carregar modelo ICM: {e}")
        return False

def generate_caption(image):
    """Gera resumo descritivo da imagem usando o modelo ICM"""
    global icm_processor, icm_model
    
    try:
        # Converter imagem OpenCV para formato PIL
        if isinstance(image, np.ndarray):
            # Converter BGR para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Processar imagem para o modelo
        inputs = icm_processor(pil_image, return_tensors="pt")
        
        # Fazer a inferência
        with torch.no_grad():
            outputs = icm_model.generate(**inputs, max_length=50)
        
        # Decodificar a saída para obter o resumo em texto
        caption = icm_processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Erro ao gerar resumo da imagem: {e}")
        return "Erro ao gerar resumo"

def get_stream_url(youtube_url):
    """Usar streamlink para obter a URL direta do stream"""
    try:
        # Executar o comando streamlink para obter apenas a URL do stream
        cmd = ["streamlink", youtube_url, "best", "--stream-url"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Erro ao obter URL do stream: {result.stderr}")
            return None
        
        # Extrair a URL do output
        stream_url = result.stdout.strip()
        if not stream_url:
            print("URL do stream vazia")
            return None
        
        print(f"URL do stream obtida com sucesso: {stream_url[:60]}...")
        return stream_url
    
    except Exception as e:
        print(f"Erro ao executar streamlink: {e}")
        return None

def load_yolo_model(config_path, weights_path):
    """Carregar modelo YOLO usando OpenCV DNN"""
    try:
        print("Carregando modelo YOLO...")
        # Carregar a rede neural
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Definir backend e target (CPU ou GPU se disponível)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Obter nomes das camadas de saída
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        print("Modelo YOLO carregado com sucesso!")
        return net, output_layers
    
    except Exception as e:
        print(f"Erro ao carregar modelo YOLO: {e}")
        return None, None

def stop_detection():
    """Parar a detecção"""
    global running, cap
    running = False
    if cap is not None:
        cap.release()

def start_image_detection():
    """Iniciar detecção em uma imagem estática"""
    global net, output_layers, classes, icm_processor, icm_model
    
    # Verificar se o modelo está carregado
    if net is None or icm_model is None:
        messagebox.showerror("Erro", "Modelos não carregados!")
        return
    
    # Abrir seletor de arquivo
    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Arquivos de imagem", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        return  # Usuário cancelou
    
    try:
        # Ler a imagem
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem!")
            return
        
        status_label.config(text=f"Status: Processando imagem {os.path.basename(file_path)}...")
        
        # Processar a imagem em uma thread separada
        threading.Thread(target=process_single_image, args=(frame, file_path), daemon=True).start()
    
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao processar a imagem: {e}")

# Nova função para carregar e processar vídeo
def start_video_detection():
    """Iniciar detecção em um arquivo de vídeo"""
    global net, output_layers, classes, icm_processor, icm_model, running, cap
    
    # Verificar se o modelo está carregado
    if net is None or icm_model is None:
        messagebox.showerror("Erro", "Modelos não carregados!")
        return
    
    # Verificar se já está rodando
    if running:
        messagebox.showinfo("Aviso", "Detecção já está em andamento!")
        return
    
    # Abrir seletor de arquivo
    file_path = filedialog.askopenfilename(
        title="Selecione um vídeo",
        filetypes=[("Arquivos de vídeo", "*.mp4 *.avi *.mkv *.mov *.wmv")]
    )
    
    if not file_path:
        return  # Usuário cancelou
    
    try:
        # Abrir o vídeo
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o arquivo de vídeo!")
            return
        
        # Iniciar thread de detecção
        running = True
        threading.Thread(target=detection_thread, daemon=True).start()
        status_label.config(text=f"Status: Processando vídeo {os.path.basename(file_path)}...")
        
        # Desabilitar botões
        start_webcam_btn.config(state=tk.DISABLED)
        start_youtube_btn.config(state=tk.DISABLED)
        start_image_btn.config(state=tk.DISABLED)
        start_video_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.NORMAL)
    
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao processar o vídeo: {e}")

def process_single_image(frame, file_path):
    """Processa uma única imagem estática"""
    global net, output_layers, classes, icm_processor, icm_model
    
    try:
        # Obter dimensões do frame
        height, width, channels = frame.shape
        
        # Preparar o blob para entrada na rede
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Passar o blob pela rede
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Informações sobre as detecções
        class_ids = []
        confidences = []
        boxes = []
        
        # Processar as detecções
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filtrar detecções fracas
                if confidence > 0.5:
                    # Coordenadas do objeto
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Coordenadas do retângulo
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # Adicionar às listas
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Aplicar Non-Maximum Suppression para remover detecções redundantes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Criar cópia para desenhar
        frame_with_boxes = frame.copy()
        
        # Contador de objetos por tipo
        obj_counts = {}
        
        # Desenhar caixas delimitadoras
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                class_name = classes[class_id]
                label = f"{class_name}: {confidences[i]:.2f}"
                
                # Contar objetos por tipo
                if class_name in obj_counts:
                    obj_counts[class_name] += 1
                else:
                    obj_counts[class_name] = 1
                
                # Definir cor com base no tipo de objeto
                if class_name == 'person':
                    color = (0, 0, 255)  # Vermelho para pessoas
                elif class_name in ['car', 'bicycle', 'truck', 'motorbike', 'bus']:
                    color = (255, 0, 0)  # Azul para veículos
                else:
                    color = (0, 255, 0)  # Verde para outros objetos
                
                # Desenhar retângulo
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), color, 2)
                
                # Desenhar rótulo
                cv2.putText(frame_with_boxes, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Gerar resumo da imagem com ICM
        caption = generate_caption(frame)
        
        # Criar sumário dos objetos detectados
        summary = "Objetos detectados:\n"
        for obj, count in obj_counts.items():
            summary += f"- {obj}: {count}\n"
        
        # Adicionar o resumo gerado pelo ICM (com tradução simples se estiver em inglês)
        if caption.strip() and caption[0].isupper():  # Verificação básica se está em inglês
            # Tentativa simples de tradução das frases mais comuns do inglês para o português
            translated_caption = caption
            english_phrases = [
                "A ", "The ", "There is ", "There are ", "This is ", "These are ",
                "It is ", "They are ", "I see ", "showing ", "with ", "and "
            ]
            portuguese_phrases = [
                "Um ", "O/A ", "Há ", "Há ", "Isto é ", "Estes são ",
                "É ", "Eles são ", "Eu vejo ", "mostrando ", "com ", "e "
            ]
            
            for i, phrase in enumerate(english_phrases):
                if phrase in translated_caption:
                    translated_caption = translated_caption.replace(phrase, portuguese_phrases[i])
            
            summary += f"\nDescrição da imagem (ICM):\n{translated_caption}\nOriginal em inglês: {caption}"
        else:
            summary += f"\nDescrição da imagem (ICM):\n{caption}"
        
        # Salvar resultados
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_folder = os.path.join(SAVE_DIR, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        
        # Salvar imagem original
        original_path = os.path.join(save_folder, "original.jpg")
        cv2.imwrite(original_path, frame)
        
        # Salvar imagem com detecções
        detection_path = os.path.join(save_folder, "detection.jpg")
        cv2.imwrite(detection_path, frame_with_boxes)
        
        # Salvar sumário em arquivo de texto
        summary_path = os.path.join(save_folder, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Atualizar interface
        def update_ui():
            # Converter imagem para exibição na interface
            frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Redimensionar mantendo a proporção
            target_width = 800
            ratio = target_width / frame_with_boxes.shape[1]
            target_height = int(frame_with_boxes.shape[0] * ratio)
            frame_pil = frame_pil.resize((target_width, target_height), Image.LANCZOS)
            
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            video_label.config(image=frame_tk)
            video_label.image = frame_tk  # Manter referência
            
            # Atualizar labels
            status_label.config(text=f"Status: Processamento concluído")
            caption_label.config(text=f"Descrição: {caption}")
            
            # Mostrar contagem de objetos
            count_text = ", ".join([f"{obj}: {count}" for obj, count in obj_counts.items()])
            count_label.config(text=f"Objetos: {count_text}")
            
            save_label.config(text=f"Resultados salvos em: {save_folder}")
        
        # Executar atualização da interface na thread principal
        root.after(0, update_ui)
    
    except Exception as e:
        def show_error():
            messagebox.showerror("Erro", "Erro ao processar a imagem: {e}")
            status_label.config(text="Status: Erro durante processamento")
        
        root.after(0, show_error)

def start_detection(source_type):
    """Iniciar a detecção de vídeo"""
    global running, cap, net, output_layers, classes
    
    if running:
        messagebox.showinfo("Aviso", "Detecção já está em andamento!")
        return
    
    # Verificar se o modelo está carregado
    if net is None or icm_model is None:
        messagebox.showerror("Erro", "Modelos não carregados!")
        return
    
    # Definir a fonte de vídeo
    if source_type == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a webcam!")
            return
    else:  # YouTube
        stream_url = get_stream_url(YOUTUBE_URL)
        if not stream_url:
            messagebox.showerror("Erro", "Não foi possível obter a URL do stream do YouTube!")
            return
        
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível conectar ao stream do YouTube!")
            return
    
    # Iniciar thread de detecção
    running = True
    threading.Thread(target=detection_thread, daemon=True).start()
    status_label.config(text="Status: Executando detecção...")
    start_webcam_btn.config(state=tk.DISABLED)
    start_youtube_btn.config(state=tk.DISABLED)
    start_image_btn.config(state=tk.DISABLED)
    start_video_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)

def detection_thread():
    """Thread principal de detecção"""
    global running, cap, net, output_layers, classes, icm_processor, icm_model
    
    # Variáveis para estatísticas
    frame_count = 0
    start_time = time.time()
    prev_fps_time = start_time
    fps = 0
    last_save_time = 0
    last_caption_time = 0
    last_caption = "Aguardando geração de resumo..."
    
    # Configuração para velocidade de reprodução
    # Isso pula frames para acelerar a reprodução (2x mais rápido)
    skip_frames = 5  # Pular 1 frame a cada 2 (reprodução 2x mais rápida)
    
    try:
        while running:
            # Ler frame
            ret, frame = cap.read()
            if not ret:
                print("Erro ao ler o frame. Tentando reconectar...")
                # Tentar reconectar
                if not running:
                    break
                # Para vídeos locais, parar a detecção quando chegar ao fim do vídeo
                if isinstance(cap, cv2.VideoCapture) and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("Fim do vídeo alcançado.")
                    running = False
                    break
                continue
                
            # Pular frames para acelerar a reprodução
            for _ in range(skip_frames):
                ret_skip, _ = cap.read()
                if not ret_skip:
                    break
            
            # Incrementar contador de frames
            frame_count += 1
            
            # Calcular FPS a cada segundo
            current_time = time.time()
            if current_time - prev_fps_time >= 1.0:
                fps = frame_count / (current_time - prev_fps_time)
                frame_count = 0
                prev_fps_time = current_time
            
            # Para melhor desempenho, processar apenas a cada 2 frames
            if frame_count % 2 != 0:
                # Mostrar estatísticas no frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Converter para exibir na GUI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Redimensionar mantendo a proporção
                target_width = 800
                ratio = target_width / frame.shape[1]
                target_height = int(frame.shape[0] * ratio)
                frame_pil = frame_pil.resize((target_width, target_height), Image.LANCZOS)
                
                frame_tk = ImageTk.PhotoImage(image=frame_pil)
                video_label.config(image=frame_tk)
                video_label.image = frame_tk  # Manter referência
                
                continue
            
            # Obter dimensões do frame
            height, width, channels = frame.shape
            
            # Preparar o blob para entrada na rede
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            
            # Passar o blob pela rede
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            # Informações sobre as detecções
            class_ids = []
            confidences = []
            boxes = []
            
            # Processar as detecções
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filtrar detecções fracas
                    if confidence > 0.5:
                        # Coordenadas do objeto
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Coordenadas do retângulo
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Adicionar às listas
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Aplicar Non-Maximum Suppression para remover detecções redundantes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Contador de objetos por tipo
            obj_counts = {}
            
            # Desenhar caixas delimitadoras
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    class_name = classes[class_id]
                    label = f"{class_name}: {confidences[i]:.2f}"
                    
                    # Contar objetos por tipo
                    if class_name in obj_counts:
                        obj_counts[class_name] += 1
                    else:
                        obj_counts[class_name] = 1
                    
                    # Definir cor com base no tipo de objeto
                    if class_name == 'person':
                        color = (0, 0, 255)  # Vermelho para pessoas
                    elif class_name in ['car', 'bicycle', 'truck', 'motorbike', 'bus']:
                        color = (255, 0, 0)  # Azul para veículos
                    else:
                        color = (0, 255, 0)  # Verde para outros objetos
                    
                    # Desenhar retângulo
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Desenhar rótulo
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Mostrar estatísticas no frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Atualizar contadores na interface
            count_text = ", ".join([f"{obj}: {count}" for obj, count in obj_counts.items()])
            count_label.config(text=f"Objetos: {count_text}")
            fps_label.config(text=f"FPS: {fps:.1f}")
            
            # Gerar resumo a cada 10 segundos
            if current_time - last_caption_time > 10:
                # Executar em uma thread separada para não travar o processamento principal
                def generate_caption_thread(frame_to_process):
                    nonlocal last_caption
                    caption = generate_caption(frame_to_process)
                    last_caption = caption
                    caption_label.config(text=f"Descrição: {caption}")
                
                threading.Thread(target=generate_caption_thread, args=(frame.copy(),), daemon=True).start()
                last_caption_time = current_time
            
            # Salvar snapshot quando objetos importantes são detectados (a cada 5 segundos)
            has_important_objects = any(obj in ['person', 'car', 'truck', 'bus'] for obj in obj_counts.keys())
            if has_important_objects and (current_time - last_save_time) > 5:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_folder = os.path.join(SAVE_DIR, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                
                # Salvar imagem com detecções
                detection_path = os.path.join(save_folder, "detection.jpg")
                cv2.imwrite(detection_path, frame)
                
                # Criar sumário dos objetos detectados
                summary = "Objetos detectados:\n"
                for obj, count in obj_counts.items():
                    summary += f"- {obj}: {count}\n"
                
                # Adicionar o resumo gerado pelo ICM
                summary += f"\nDescrição da imagem (ICM):\n{last_caption}"
                
                # Salvar sumário em arquivo de texto (em português)
                summary_path = os.path.join(save_folder, "resumo.txt")
                
                # Traduzir automaticamente para português se o resumo estiver em inglês
                # (O modelo ICM geralmente gera descrições em inglês)
                portugues_summary = summary.replace("Objects detected:", "Objetos detectados:")
                if "Description" in portugues_summary:
                    portugues_summary = portugues_summary.replace("Description", "Descrição")
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(portugues_summary)
                
                print(f"Detecção salva: {detection_path}")
                save_label.config(text=f"Última detecção salva: {detection_path}")
                last_save_time = current_time
                
                # Mostrar indicador de gravação
                cv2.putText(frame, "Salvando detecção!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Converter para exibir na GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Redimensionar mantendo a proporção
            target_width = 800
            ratio = target_width / frame.shape[1]
            target_height = int(frame.shape[0] * ratio)
            frame_pil = frame_pil.resize((target_width, target_height), Image.LANCZOS)
            
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            video_label.config(image=frame_tk)
            video_label.image = frame_tk  # Manter referência
    
    except Exception as e:
        print(f"Erro durante processamento: {e}")
    finally:
        if cap is not None:
            cap.release()
        
        # Restaurar interface
        def restore_ui():
            status_label.config(text="Status: Detecção parada")
            start_webcam_btn.config(state=tk.NORMAL)
            start_youtube_btn.config(state=tk.NORMAL)
            start_image_btn.config(state=tk.NORMAL)
            start_video_btn.config(state=tk.NORMAL)
            stop_btn.config(state=tk.DISABLED)
        
        if threading.current_thread() != threading.main_thread():
            root.after(0, restore_ui)
        else:
            restore_ui()

def initialize_models():
    """Inicializar o modelo YOLO e ICM"""
    global net, output_layers, classes, icm_processor, icm_model
    
    status_label.config(text="Status: Inicializando modelos...")
    
    # Função para executar em segundo plano
    def init_thread():
        global net, output_layers, classes, icm_processor, icm_model
        
        # Baixar e carregar arquivos YOLO
        success, config_path, weights_path, classes_list = download_yolo_files()
        if not success:
            messagebox.showerror("Erro", "Não foi possível baixar ou localizar os arquivos do modelo YOLO.")
            status_label.config(text="Status: Erro ao inicializar YOLO")
            return
        
        # Carregar modelo YOLO
        net, output_layers = load_yolo_model(config_path, weights_path)
        if net is None:
            messagebox.showerror("Erro", "Não foi possível carregar o modelo YOLO.")
            status_label.config(text="Status: Erro ao carregar modelo YOLO")
            return
        
        classes = classes_list
        
        # Inicializar modelo ICM
        if not initialize_icm_model():
            messagebox.showerror("Erro", "Não foi possível carregar o modelo ICM (Gerador de Resumos).")
            status_label.config(text="Status: Erro ao carregar modelo ICM")
            return
        
        status_label.config(text="Status: Modelos inicializados com sucesso!")
        
        # Habilitar botões
        start_webcam_btn.config(state=tk.NORMAL)
        start_youtube_btn.config(state=tk.NORMAL)
        start_image_btn.config(state=tk.NORMAL)
        start_video_btn.config(state=tk.NORMAL)
    
    # Iniciar thread
    threading.Thread(target=init_thread, daemon=True).start()

def on_closing():
    """Função chamada ao fechar a janela"""
    global running
    if running:
        stop_detection()
    root.destroy()

# Criar interface gráfica
root = tk.Tk()
root.title("Sistema Integrado YOLO-ICM para Detecção e Resumo de Imagens")
root.geometry("900x800")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Área principal para exibição do vídeo
video_frame = ttk.Frame(root)
video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

video_label = ttk.Label(video_frame)
video_label.pack(fill=tk.BOTH, expand=True)

# Área de controles
control_frame = ttk.Frame(root)
control_frame.pack(padx=10, pady=5, fill=tk.X)

# Botões
btn_frame = ttk.Frame(control_frame)
btn_frame.pack(pady=5)

start_webcam_btn = ttk.Button(btn_frame, text="Iniciar Webcam", command=lambda: start_detection("webcam"), state=tk.DISABLED)
start_webcam_btn.grid(row=0, column=0, padx=5)

start_youtube_btn = ttk.Button(btn_frame, text="Iniciar YouTube", command=lambda: start_detection("youtube"), state=tk.DISABLED)
start_youtube_btn.grid(row=0, column=1, padx=5)

start_image_btn = ttk.Button(btn_frame, text="Carregar Imagem", command=start_image_detection, state=tk.DISABLED)
start_image_btn.grid(row=0, column=2, padx=5)

# Novo botão para carregar vídeo
start_video_btn = ttk.Button(btn_frame, text="Carregar Vídeo", command=start_video_detection, state=tk.DISABLED)
start_video_btn.grid(row=0, column=3, padx=5)

stop_btn = ttk.Button(btn_frame, text="Parar", command=stop_detection, state=tk.DISABLED)
stop_btn.grid(row=0, column=4, padx=5)

# Status
status_frame = ttk.Frame(control_frame)
status_frame.pack(pady=5, fill=tk.X)

status_label = ttk.Label(status_frame, text="Status: Aguardando inicialização...")
status_label.pack(anchor=tk.W)

count_label = ttk.Label(status_frame, text="Objetos: -")
count_label.pack(anchor=tk.W)

fps_label = ttk.Label(status_frame, text="FPS: 0.0")
fps_label.pack(anchor=tk.W)

caption_label = ttk.Label(status_frame, text="Descrição: Aguardando geração de resumo...", wraplength=850)
caption_label.pack(anchor=tk.W, pady=5)

save_label = ttk.Label(status_frame, text="Última detecção salva: -")
save_label.pack(anchor=tk.W)

# Inicializar modelos
initialize_models()

# Iniciar loop principal
if __name__ == "__main__":
    root.mainloop()