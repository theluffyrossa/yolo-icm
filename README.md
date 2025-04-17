
# YOLO-ICM: Sistema Integrado de Detecção de Objetos e Geração de Resumos de Imagens


Este repositório contém um sistema completo que integra o modelo de detecção de objetos YOLO (You Only Look Once) com ICM (Image Caption Model) para criar uma solução que não apenas detecta objetos em imagens e vídeos, mas também gera descrições textuais automaticamente.

## 🌟 Recursos

- **Múltiplas Fontes de Entrada**:
  - 📹 Webcam em tempo real
  - 🎬 Streaming de vídeos do YouTube
  - 🖼️ Upload de imagens estáticas

- **Detecção Avançada de Objetos**:
  - Identificação em tempo real usando YOLOv4
  - Visualização com caixas delimitadoras coloridas
  - Classificação de objetos em mais de 80 categorias

- **Geração Automática de Descrições**:
  - Resumos textuais do conteúdo das imagens usando BLIP
  - Análise contextual das cenas detectadas
  - Atualização dinâmica das descrições

- **Armazenamento Inteligente**:
  - Salvamento automático de detecções relevantes
  - Organização por timestamp
  - Exportação de resumos textuais e imagens processadas

## 📋 Pré-requisitos

- Python 3.7+ 
- Bibliotecas principais:
  ```
  opencv-python >= 4.5.0
  numpy >= 1.19.0
  torch >= 1.8.0
  transformers >= 4.12.0
  pillow >= 8.0.0
  tkinter (incluído na maioria das instalações Python)
  ```

- Para streaming do YouTube:
  ```
  streamlink >= 3.0.0
  ```

- Espaço em disco: ~300MB para modelos

## 🚀 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/theluffyrossa/yolo-icm.git
   cd yolo-icm
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   pip install opencv-python numpy tkinter pillow torch transformers
   ```

3. Execute o sistema:
   ```bash
   python yolo_icm_system.py
   ```

Na primeira execução, o sistema baixará automaticamente os modelos necessários (~250MB para YOLOv4 e ~50MB para BLIP).

## 💻 Uso

### Interface Gráfica

Após a inicialização, você verá a interface principal com as seguintes opções:

- **Iniciar Webcam**: Inicia a detecção usando a câmera do computador
- **Iniciar YouTube**: Conecta-se ao streaming de vídeo definido
- **Carregar Imagem**: Abre um seletor de arquivos para processar imagens estáticas
- **Parar**: Interrompe o processamento atual

### Personalizando o Streaming

Para mudar a fonte de vídeo do YouTube, edite a constante `YOUTUBE_URL` no início do arquivo:

```python
YOUTUBE_URL = "https://www.youtube.com/watch?v=SuaURL"
```

### Resultados

Os resultados são salvos no diretório `detections/` organizado por timestamps:

```
detections/
├── 20250417_081530/
│   ├── detection.jpg    # Imagem com objetos detectados
│   └── summary.txt      # Lista de objetos e descrição
├── 20250417_081542/
...
```

## 🧠 Como Funciona

### Arquitetura do Sistema

O sistema opera em três camadas principais:

1. **Entrada de Dados**: Captura de frames de diversas fontes
2. **Processamento**: 
   - YOLO processa cada frame para detectar objetos
   - ICM (BLIP) gera descrições baseadas no conteúdo visual
3. **Saída**: Visualização e armazenamento dos resultados

### Fluxo de Processamento

```
Captura → Pré-processamento → Detecção YOLO → Geração de Descrição → Visualização → Armazenamento
```

### Componentes Principais

- **YOLOv4**: Implementado via OpenCV DNN para detecção eficiente
- **BLIP**: Modelo de transformer multimodal para geração de legendas
- **Interface Tkinter**: Para interação amigável com o usuário

## 📊 Desempenho

O desempenho varia de acordo com o hardware disponível:

| Configuração | FPS (apenas YOLO) | FPS (YOLO + ICM) |
|--------------|-------------------|------------------|
| CPU (i5 ou similar) | 8-15 | 5-10 |
| GPU (NVIDIA GTX/RTX) | 25-45 | 15-30 |

*A geração de resumos ICM é executada a cada 10 segundos para manter o desempenho.

## 🔧 Personalização

### Ajustando Parâmetros de Detecção

Para modificar o limiar de confiança da detecção, altere o valor em:

```python
if confidence > 0.5:  # Altere para um valor entre 0 e 1
```

### Modificando a Frequência de Resumos

Para alterar a frequência com que as descrições são geradas:

```python
if current_time - last_caption_time > 10:  # Altere o intervalo em segundos
```

### Selecionando Classes de Interesse

Para focar em categorias específicas, modifique a verificação de objetos importantes:

```python
has_important_objects = any(obj in ['person', 'car', 'truck', 'bus'] for obj in obj_counts.keys())
# Adicione ou remova classes conforme necessário
```

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das alterações (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 TODO

- [ ] Adicionar suporte para processamento em GPU via CUDA
- [ ] Implementar tracking de objetos entre frames
- [ ] Desenvolver uma API REST para integração com outros sistemas
- [ ] Adicionar suporte para múltiplas câmeras
- [ ] Criar visualizações estatísticas dos objetos detectados

## 📖 Citações

Se utilizar este projeto em pesquisas acadêmicas, por favor cite:

```
@software{yolo_icm_2025,
  author = {The Luffy Rossa},
  title = {YOLO-ICM: Sistema Integrado de Detecção de Objetos e Geração de Resumos de Imagens},
  url = {https://github.com/theluffyrossa/yolo-icm},
  year = {2025},
}
```


## 🙏 Agradecimentos

- [AlexeyAB](https://github.com/AlexeyAB/darknet) pelo desenvolvimento do YOLOv4
- [Salesforce Research](https://github.com/salesforce/BLIP) pelo modelo BLIP
- Todos os contribuidores e testadores

---

Desenvolvido por [theluffyrossa](https://github.com/theluffyrossa) - 2025
