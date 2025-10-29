import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re
import os
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model.eval()


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
    # Загружаем теги из файлов
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
    analyze_pdf("fizika_10kl_gromika_rus_2019.pdf")  # Замените путь
