import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_metrics import TRUEslatorMetrics
from PIL import Image
import numpy as np
from ultralytics import YOLO
from utils.predict_bounding_boxes import predict_bounding_boxes
from utils.translate_manga import translate_manga
from utils.manga_ocr_utils import get_text_from_image
from utils.process_contour import process_contour
from utils.write_text_on_image import add_text

def process_image(image_path, model):
    """Processa uma imagem usando o pipeline real do TRUEslator"""
    # Carregar a imagem
    image = np.array(Image.open(image_path))
    original_image = image.copy()
    
    # Prever caixas delimitadoras
    results = predict_bounding_boxes(model, image_path)
    predicted_boxes = []
    extracted_texts = []
    translated_texts = []
    
    for result in results:
        # Descompacta as coordenadas e outras informações da detecção
        x1, y1, x2, y2, score, class_id = result
        predicted_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        detected_image = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Converte a imagem detectada para o formato PIL
        im = Image.fromarray(np.uint8(detected_image * 255))
        
        # Extrai o texto da imagem
        text = get_text_from_image(im)
        extracted_texts.append(text)
        
        # Processa os contornos da imagem
        detected_image, cont = process_contour(detected_image)
        
        # Traduz o texto extraído
        text_translated = translate_manga(text, source_lang='auto', target_lang='en')
        translated_texts.append(text_translated)
        
        # Adiciona o texto traduzido na imagem detectada
        image_with_text = add_text(detected_image, text_translated, cont)
        
        # Substitui a região da imagem original com a versão modificada
        image[int(y1):int(y2), int(x1):int(x2)] = image_with_text
    
    # Converte a imagem final para PIL
    result_image = Image.fromarray(image, 'RGB')
    
    return {
        'predicted_boxes': predicted_boxes,
        'translated_texts': translated_texts,
        'extracted_texts': extracted_texts,
        'result_image': result_image,
        'original_image': Image.fromarray(original_image, 'RGB')
    }

def load_example_data():
    """Carrega os dados de exemplo para avaliação usando o pipeline real"""
    # Carregar modelo de detecção de objetos
    best_model_path = "model_creation/runs/detect/train5/"
    object_detection_model = YOLO(os.path.join(best_model_path, "weights/best.pt"))
    
    # Processar imagem de exemplo
    example_image_path = 'validation_example_pages/1.jpg'
    processed_data = process_image(example_image_path, object_detection_model)
    
    # Carregar textos original e de referência
    with open('ground_truth/raw/1.txt', 'r', encoding='utf-8') as f:
        original_text = f.read().strip()
    
    with open('ground_truth/translated/1.txt', 'r', encoding='utf-8') as f:
        reference_text = f.read().strip()
    
    # Combinar todos os textos traduzidos em um único texto
    translated_text = ' '.join(processed_data['translated_texts'])
    extracted_text = ' '.join(processed_data['extracted_texts'])
    
    # Carregar ground truth boxes (em um caso real, isso seria anotado manualmente)
    ground_truth_boxes = [(105, 105, 195, 145), (305, 205, 395, 245)]
    
    return {
        'original_text': original_text,
        'extracted_text': extracted_text,
        'translated_text': translated_text,
        'reference_text': reference_text,
        'original_img': processed_data['original_image'],
        'inpainted_img': processed_data['result_image'],
        'predicted_boxes': processed_data['predicted_boxes'],
        'ground_truth_boxes': ground_truth_boxes
    }

def main():
    # Inicializar o avaliador de métricas
    evaluator = TRUEslatorMetrics()
    
    # Carregar dados usando o pipeline real
    data = load_example_data()
    
    # Salvar textos para análise manual
    os.makedirs('metrics/comparison', exist_ok=True)
    
    # Salvar texto extraído pelo OCR
    with open('metrics/comparison/extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(data['extracted_text'])
    
    # Salvar texto traduzido
    with open('metrics/comparison/translated_text.txt', 'w', encoding='utf-8') as f:
        f.write(data['translated_text'])
    
    # Salvar texto de referência
    with open('metrics/comparison/reference_text.txt', 'w', encoding='utf-8') as f:
        f.write(data['reference_text'])
    
    # Avaliar e salvar métricas
    metrics = evaluator.evaluate_and_save(
        original_text=data['original_text'],
        translated_text=data['translated_text'],
        reference_text=data['reference_text'],
        original_img=data['original_img'],
        inpainted_img=data['inpainted_img'],
        predicted_boxes=data['predicted_boxes'],
        ground_truth_boxes=data['ground_truth_boxes']
    )
    
    # Exibir resultados
    print('\nResultados da Avaliação:')
    print(f"Qualidade da tradução: {metrics['translation_quality']:.4f}")
    print(f"Qualidade do Inpainting: {metrics['inpainting_quality']:.4f}")
    print(f"Taxa de Detecção: {metrics['text_detection_rate']:.4f}")
    print(f"Pontuação Geral: {metrics['overall_score']:.4f}")
    print(f"\nGráfico salvo em: {evaluator.plot_file}")
    print(f"\nTextos salvos para análise manual em metrics/comparison/")
    print("- Texto extraído: extracted_text.txt")
    print("- Texto traduzido: translated_text.txt")
    print("- Texto de referência: reference_text.txt")

if __name__ == '__main__':
    main()