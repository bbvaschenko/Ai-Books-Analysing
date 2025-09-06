import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re


tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
model.eval()



section_tags = [  
    "алгебра", "геометрия", "дифференциальные уравнения", "теория вероятностей",
    "численные методы", "математическая логика", "линейная алгебра", "математический анализ",
    "топология", "механика", "оптика", "термодинамика", "электродинамика",
    "программирование", "машинное обучение", "вычислительная математика",
    "дискретная математика", "комбинаторика", "теория графов", "криптография"
]

section_to_area = {
    "алгебра": "математика",
    "геометрия": "математика",
    "дифференциальные уравнения": "математика",
    "теория вероятностей": "математика",
    "численные методы": "математика",
    "математическая логика": "математика",
    "линейная алгебра": "математика",
    "математический анализ": "математика",
    "топология": "математика",
    "механика": "физика",
    "оптика": "физика",
    "термодинамика": "физика",
    "электродинамика": "физика",
    "программирование": "информатика",
    "машинное обучение": "информатика",
    "вычислительная математика": "информатика",
    "дискретная математика": "информатика",
    "комбинаторика": "математика",
    "теория графов": "математика",
    "криптография": "информатика"
}

concept_tags = [  
    "матрица", "интеграл", "дифференциал", "уравнение", "доказательство",
    "логика", "формула", "вектор", "градиент", "предел", "ряды",
    "частная производная", "функция", "оператор", "конвергенция",
    "расходимость", "дисперсия", "среднее значение", "корень", "производная"
]

difficulty_tags = [  
    "начальный уровень", "средний уровень", "продвинутый уровень", "базовый курс",
    "университетский курс", "магистратура", "аспирантура", "вводный материал",
    "теоретический уровень", "прикладной уровень"
]

# === ФУНКЦИИ ===

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_embeddings(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(emb)

    return torch.mean(torch.stack(embeddings), dim=0)

def get_tags(text_embedding, tag_list, top_k=5):
    tag_embeddings = []
    for tag in tag_list:
        inputs = tokenizer(tag, return_tensors='pt', truncation=True, padding=True, max_length=10)
        with torch.no_grad():
            outputs = model(**inputs)
        tag_embedding = outputs.last_hidden_state.mean(dim=1)
        tag_embeddings.append(tag_embedding)

    tag_embeddings = torch.cat(tag_embeddings)
    similarities = F.cosine_similarity(text_embedding, tag_embeddings)
    top_indices = similarities.topk(top_k).indices
    return [tag_list[i] for i in top_indices]

def infer_area_from_sections(section_tags_found):
    area_counter = {}
    for section in section_tags_found:
        area = section_to_area.get(section)
        if area:
            area_counter[area] = area_counter.get(area, 0) + 1

    if area_counter:
        sorted_areas = sorted(area_counter.items(), key=lambda x: -x[1])
        return [sorted_areas[0][0]]
    return ["не определено"]


def analyze_pdf(pdf_path):
    print("Извлечение текста из PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text or len(raw_text.strip()) < 200:
        print("Недостаточно текста для анализа.")
        return

    print("Генерация эмбеддингов по всему тексту...")
    text_embedding = extract_embeddings(raw_text)

    print("Анализ тегов...")
    found_sections = get_tags(text_embedding, section_tags, top_k=3)
    found_area = infer_area_from_sections(found_sections)
    found_concepts = get_tags(text_embedding, concept_tags, top_k=3)
    found_difficulty = get_tags(text_embedding, difficulty_tags, top_k=2)

    print("\nРезультаты анализа:")
    print(f"Область знаний: {', '.join(found_area)}")
    print(f"Разделы: {', '.join(found_sections)}")
    print(f"Термины и понятия: {', '.join(found_concepts)}")
    print(f"Сложность материала: {', '.join(found_difficulty)}")



if __name__ == "__main__":
    analyze_pdf("example.pdf")  
