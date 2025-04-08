import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from skimage.metrics import structural_similarity as ssim
import json
from datetime import datetime
import os
import difflib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TRUEslatorMetrics:
    def __init__(self):
        self.metrics_file = "metrics/reports/metrics_history.json"
        self.plot_file = "metrics/plots/metrics_evolution.png"
        self._ensure_directories()
        self.metrics_history = self._load_metrics_history()
        # Inicializa o modelo de embeddings
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    def _ensure_directories(self):
        """Garante que os diretórios necessários existam"""
        os.makedirs("metrics/reports", exist_ok=True)
        os.makedirs("metrics/plots", exist_ok=True)

    def _load_metrics_history(self):
        """Carrega o histórico de métricas do arquivo JSON"""
        try:
            with open(self.metrics_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def calculate_semantic_similarity(self, text1, text2):
        """Calcula a similaridade semântica entre dois textos usando embeddings"""
        # Gera embeddings para os textos
        embedding1 = self.embedding_model.encode([text1])
        embedding2 = self.embedding_model.encode([text2])

        # Calcula similaridade de cosseno
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return max(0, similarity)  # Normaliza para [0, 1]

    def calculate_translation_quality(self, translated_text, reference_text):
        """Calcula a qualidade da tradução usando similaridade semântica baseada em embeddings"""
        # Normaliza os textos
        translated_text = translated_text.lower().strip()
        reference_text = reference_text.lower().strip()

        # Calcula similaridade semântica usando embeddings
        return self.calculate_semantic_similarity(translated_text, reference_text)

    def calculate_inpainting_quality(self, original_img, inpainted_img):
        """Calcula a qualidade do inpainting usando SSIM"""
        # Converter imagens para arrays numpy e garantir mesmo tamanho
        original_array = np.array(original_img.convert("L"))
        inpainted_array = np.array(inpainted_img.convert("L").resize(original_img.size))

        # Calcular SSIM
        score, _ = ssim(original_array, inpainted_array, full=True)
        return max(0, score)  # Normalizar para [0, 1]

    def calculate_text_detection_rate(self, reference_text, extracted_text):
        """Calcula a taxa de detecção usando similaridade de caracteres e análise léxica"""
        if not reference_text or not extracted_text:
            return 0.0

        # Normalização e preparação dos textos
        # Carregar texto original japonês para comparação
        with open("ground_truth/raw/1.txt", "r", encoding="utf-8") as f:
            original_jp = f.read().lower().strip()

        # Comparar OCR com original japonês
        ocr = extracted_text.lower().strip()
        ref = original_jp

        # 1. Similaridade de caracteres (exata e aproximada)
        char_match = sum(1 for a, b in zip(ref, ocr) if a == b) / max(
            len(ref), len(ocr)
        )

        # 2. Análise de frequência léxica
        ref_words = set(ref.split())
        ocr_words = set(ocr.split())
        lexical_overlap = (
            len(ref_words & ocr_words) / len(ref_words) if ref_words else 0
        )

        # 3. Detecção de erros comuns de OCR
        common_errors = {"0": "o", "1": "i", "5": "s", "8": "b", "@": "a"}
        corrected_ocr = "".join([common_errors.get(c, c) for c in ocr])
        error_corrected_match = sum(
            1 for a, b in zip(ref, corrected_ocr) if a == b
        ) / max(len(ref), len(corrected_ocr))

        # 4. Alinhamento de sequência usando Levenshtein
        seq_matcher = difflib.SequenceMatcher(None, ref, ocr)
        similarity_ratio = seq_matcher.ratio()

        # Cálculo combinado com pesos
        weights = {
            "char_match": 0.4,
            "lexical_overlap": 0.3,
            "error_corrected": 0.2,
            "levenshtein": 0.1,
        }

        final_score = (
            char_match * weights["char_match"]
            + lexical_overlap * weights["lexical_overlap"]
            + error_corrected_match * weights["error_corrected"]
            + similarity_ratio * weights["levenshtein"]
        )

        return max(0, min(final_score, 1))  # Garante valor entre 0-1

    def calculate_overall_score(self, metrics):
        """Calcula a pontuação geral combinando todas as métricas"""
        weights = {
            "translation_quality": 0.4,
            "inpainting_quality": 0.3,
            "text_detection_rate": 0.3,
        }

        return sum(metrics[key] * weights[key] for key in weights)

    def _convert_to_native_types(self, obj):
        """Converte valores numpy para tipos nativos do Python"""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {
                key: self._convert_to_native_types(value) for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(item) for item in obj]
        return obj

    def evaluate_and_save(
        self,
        original_text,
        translated_text,
        reference_text,
        original_img,
        inpainted_img,
        predicted_boxes=None,
        ground_truth_boxes=None,
    ):
        """Avalia todas as métricas e salva os resultados"""
        # Calcular métricas individuais
        metrics = {
            "translation_quality": self.calculate_translation_quality(
                translated_text, reference_text
            ),
            "inpainting_quality": self.calculate_inpainting_quality(
                original_img, inpainted_img
            ),
            "text_detection_rate": self.calculate_text_detection_rate(
                reference_text, original_text
            ),
        }

        # Calcular pontuação geral
        metrics["overall_score"] = self.calculate_overall_score(metrics)

        # Adicionar timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        # Converter valores numpy para tipos nativos do Python
        metrics = self._convert_to_native_types(metrics)

        # Atualizar histórico
        self.metrics_history.append(metrics)

        # Salvar histórico atualizado
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Gerar gráfico de evolução
        self._plot_metrics_evolution()

        return metrics

    def _plot_metrics_evolution(self):
        """Gera um gráfico mostrando a evolução das métricas ao longo do tempo"""
        if not self.metrics_history:
            return

        # Preparar dados para o gráfico
        timestamps = range(len(self.metrics_history))
        metrics_data = {
            "Qualidade da Tradução": [
                m["translation_quality"] for m in self.metrics_history
            ],
            "Qualidade do Inpainting": [
                m["inpainting_quality"] for m in self.metrics_history
            ],
            "Taxa de Detecção": [
                m["text_detection_rate"] for m in self.metrics_history
            ],
            "Pontuação Geral": [m["overall_score"] for m in self.metrics_history],
        }

        # Criar gráfico
        plt.figure(figsize=(12, 6))
        for metric_name, values in metrics_data.items():
            plt.plot(timestamps, values, marker="o", label=metric_name)

        plt.title("Evolução das Métricas do TRUEslator")
        plt.xlabel("Versão")
        plt.ylabel("Pontuação")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(self.plot_file)
        plt.close()
