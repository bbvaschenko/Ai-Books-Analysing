import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re
import os
import pandas as pd

# === МОДЕЛЬ ===
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model.eval()


# === ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ТЕГОВ ИЗ ФАЙЛОВ ===
def load_tags_from_files(tags_directory="tags"):
    tags_dict = {}

    if not os.path.exists(tags_directory):
        os.makedirs(tags_directory)
        print(f"Создана директория {tags_directory}. Добавьте туда текстовые файлы с тегами.")
        return tags_dict

    for filename in os.listdir(tags_directory):
        if filename.endswith(".txt"):
            category = filename[:-4]  # убираем расширение .txt
            filepath = os.path.join(tags_directory, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                tags = [line.strip() for line in f if line.strip()]
                tags_dict[category] = tags
                print(f"Загружено {len(tags)} тегов из категории '{category}'")

    return tags_dict


def load_area_mapping(mapping_file="tags/области_знаний.txt"):
    section_to_area = {}

    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    section, area = line.split(':', 1)
                    section_to_area[section.strip()] = area.strip()
        print(f"Загружено {len(section_to_area)} соответствий разделов и областей")
    else:
        print(f"Файл с соответствиями {mapping_file} не найден")

    return section_to_area


# === ФУНКЦИИ ДЛЯ ОПРЕДЕЛЕНИЯ АВТОРА ===
def extract_author_from_text(text):
    patterns = [
        r'Автор[:\s]+([^\n,.]+(?:\s+[^\n,.]+){0,3})',
        r'Авторы[:\s]+([^\n,.]+(?:\s+[^\n,.]+){0,5})',
        r'©\s*[^,\n]+\s*,\s*([^,\n]+)',
        r'[А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+',  # ФИО из трех слов
        r'[А-Я][а-я]+\s+[А-Я]\.[А-Я]\.',  # Имя + инициалы
    ]

    beginning_text = text[:2000].lower()

    for pattern in patterns:
        matches = re.findall(pattern, beginning_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            author = matches[0].strip()
            author = re.sub(r'^(автор|авторы|под ред|ред\.|сост\.)\s*[:\-]?\s*', '', author, flags=re.IGNORECASE)
            if author and len(author) > 3:
                return author.capitalize()

    return None


def extract_author_from_filename(filename):
    name_without_ext = os.path.splitext(filename)[0]

    patterns = [
        r'^([^\-]+)\s*-\s*',  # Автор - Название
        r'^([^_]+)_',  # Автор_Название
        r'^([^,]+),\s*',  # Автор, Название
    ]

    for pattern in patterns:
        match = re.match(pattern, name_without_ext)
        if match:
            author = match.group(1).strip()
            if re.search(r'[а-яa-z]+\s+[а-яa-z]+', author, re.IGNORECASE):
                return author

    return None


def find_author(pdf_path, text):
    author_from_text = extract_author_from_text(text)
    if author_from_text:
        return author_from_text

    filename = os.path.basename(pdf_path)
    author_from_filename = extract_author_from_filename(filename)
    if author_from_filename:
        return author_from_filename

    return "Не определен"


# === ОСНОВНЫЕ ФУНКЦИИ ===

def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)


def extract_embeddings(text, chunk_size=512):
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
    if not tag_list:
        return []

    tag_embeddings = []
    for tag in tag_list:
        inputs = tokenizer(tag, return_tensors='pt', truncation=True, padding=True, max_length=10)
        with torch.no_grad():
            outputs = model(**inputs)
        tag_embedding = outputs.last_hidden_state.mean(dim=1)
        tag_embeddings.append(tag_embedding)

    tag_embeddings = torch.cat(tag_embeddings)
    similarities = F.cosine_similarity(text_embedding, tag_embeddings)
    top_indices = similarities.topk(min(top_k, len(tag_list))).indices
    return sorted([tag_list[i] for i in top_indices])


def infer_area_from_sections(section_tags_found, section_to_area):
    area_counter = {}
    for section in section_tags_found:
        area = section_to_area.get(section)
        if area:
            area_counter[area] = area_counter.get(area, 0) + 1

    if area_counter:
        sorted_areas = sorted(area_counter.items(), key=lambda x: -x[1])
        return [sorted_areas[0][0]]
    return ["не определено"]


def get_next_book_number(excel_file):
    if not os.path.exists(excel_file):
        return 1
    df = pd.read_excel(excel_file)
    if df.empty:
        return 1
    return df['Номер книги'].max() + 1


def save_to_excel(excel_file, book_data):
    df = pd.DataFrame([book_data])
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_excel(excel_file, index=False)


# === ГЛАВНАЯ ФУНКЦИЯ ===

def analyze_pdf(pdf_path, excel_file="analyzed_books.xlsx"):
    tags_dict = load_tags_from_files()
    section_to_area = load_area_mapping()

    if not tags_dict:
        print("Не найдены файлы с тегами. Создайте директорию 'tags' и добавьте файлы с тегами.")
        return

    print("Извлечение текста из PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text or len(raw_text.strip()) < 200:
        print("Недостаточно текста для анализа.")
        return

    print("Определение автора...")
    author = find_author(pdf_path, raw_text)

    print("Генерация эмбеддингов по тексту...")
    text_embedding = extract_embeddings(raw_text)

    print("Анализ тегов...")

    found_tags = {}
    for category, tags in tags_dict.items():
        if category == "области_знаний":
            continue
        found_tags[category] = get_tags(text_embedding, tags, top_k=3)
        print(f"{category}: {', '.join(found_tags[category])}")

    section_tags = tags_dict.get("разделы", [])
    found_area = infer_area_from_sections(found_tags.get("разделы", []), section_to_area)

    book_number = get_next_book_number(excel_file)
    book_id = f"{book_number:04d}"

    book_data = {
        "Номер книги": book_number,
        "ID книги": book_id,
        "Имя файла": os.path.basename(pdf_path),
        "Автор": author,
        "Область знаний": ', '.join(found_area),
    }

    for category in tags_dict.keys():
        if category != "области_знаний":
            book_data[category.capitalize()] = ', '.join(found_tags.get(category, []))

    print("\nРезультаты анализа:")
    for key, value in book_data.items():
        print(f"{key}: {value}")

    save_to_excel(excel_file, book_data)
    print(f"\nДанные сохранены в файл: {excel_file}")

if __name__ == "__main__":
    analyze_pdf("besov.pdf")  # Замените путь
