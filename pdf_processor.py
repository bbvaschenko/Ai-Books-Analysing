import PyPDF2
import logging
import os
from typing import Dict, Any, Optional
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Класс для обработки PDF файлов.
    Извлекает текст и метаданные.
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def extract_text(self, pdf_path: str, max_pages: int = 100) -> Dict[str, Any]:
        """
        Извлечение текста из PDF файла.
        
        Args:
            pdf_path: Путь к PDF файлу
            max_pages: Максимальное количество страниц для обработки
            
        Returns:
            Словарь с текстом и метаданными
        """
        try:
            if not os.path.exists(pdf_path):
                return {"error": f"Файл не найден: {pdf_path}"}
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Извлечение метаданных
                metadata = pdf_reader.metadata or {}
                
                # Извлечение текста
                text = ""
                total_pages = len(pdf_reader.pages)
                pages_to_process = min(total_pages, max_pages)
                
                for page_num in range(pages_to_process):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n\n--- Страница {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Ошибка чтения страницы {page_num}: {e}")
                        continue
                
                result = {
                    "success": True,
                    "file_path": pdf_path,
                    "file_size": os.path.getsize(pdf_path),
                    "total_pages": total_pages,
                    "processed_pages": pages_to_process,
                    "metadata": {
                        "title": metadata.get('/Title', ''),
                        "author": metadata.get('/Author', ''),
                        "subject": metadata.get('/Subject', ''),
                        "creator": metadata.get('/Creator', ''),
                        "producer": metadata.get('/Producer', ''),
                        "creation_date": metadata.get('/CreationDate', ''),
                        "modification_date": metadata.get('/ModDate', '')
                    },
                    "text": text.strip(),
                    "text_length": len(text),
                    "estimated_words": len(text.split()),
                    "processing_timestamp": PyPDF2.__version__
                }
                
                logger.info(f"Извлечен текст из {pdf_path}: {result['text_length']} символов")
                return result
                
        except Exception as e:
            logger.error(f"Ошибка обработки PDF {pdf_path}: {e}")
            return {
                "error": str(e),
                "file_path": pdf_path,
                "success": False
            }
    
    def extract_specific_pages(self, pdf_path: str, page_numbers: list) -> Dict[str, Any]:
        """
        Извлечение текста с конкретных страниц.
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text = ""
                for page_num in page_numbers:
                    if 0 <= page_num < len(pdf_reader.pages):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        text += f"\n\n--- Страница {page_num + 1} ---\n{page_text}"
                
                return {
                    "success": True,
                    "pages_extracted": len(page_numbers),
                    "text": text.strip(),
                    "text_length": len(text)
                }
                
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Получение информации о PDF файле без извлечения текста.
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = {
                    "file_path": pdf_path,
                    "file_size": os.path.getsize(pdf_path),
                    "total_pages": len(pdf_reader.pages),
                    "is_encrypted": pdf_reader.is_encrypted,
                    "metadata": dict(pdf_reader.metadata) if pdf_reader.metadata else {},
                    "outline": len(pdf_reader.outline) if hasattr(pdf_reader, 'outline') else 0
                }
                
                return info
                
        except Exception as e:
            return {"error": str(e)}
    
    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Валидация PDF файла.
        """
        try:
            with open(pdf_path, 'rb') as file:
                # Проверка сигнатуры PDF
                header = file.read(5)
                is_pdf = header == b'%PDF-'
                
                if not is_pdf:
                    return {"valid": False, "reason": "Неверная сигнатура PDF"}
                
                # Попытка чтения
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                return {
                    "valid": True,
                    "page_count": page_count,
                    "readable": page_count > 0,
                    "encrypted": pdf_reader.is_encrypted
                }
                
        except Exception as e:
            return {"valid": False, "reason": str(e)}
    
    def batch_process(self, pdf_files: list, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Пакетная обработка нескольких PDF файлов.
        """
        results = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "results": [],
            "total_text_length": 0
        }
        
        for pdf_file in pdf_files:
            try:
                result = self.extract_text(pdf_file)
                
                if result.get("success"):
                    results["successful"] += 1
                    results["total_text_length"] += result.get("text_length", 0)
                    
                    # Сохранение в файл если указана директория
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        base_name = os.path.basename(pdf_file).replace('.pdf', '.txt')
                        output_path = os.path.join(output_dir, base_name)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(result.get("text", ""))
                        
                        result["text_file"] = output_path
                
                else:
                    results["failed"] += 1
                
                results["results"].append(result)
                
            except Exception as e:
                results["failed"] += 1
                results["results"].append({
                    "file": pdf_file,
                    "error": str(e),
                    "success": False
                })
        
        return results


# Пример использования
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Тестирование
    test_file = "test.pdf"  # Замените на реальный файл
    
    if os.path.exists(test_file):
        # Извлечение информации
        info = processor.get_pdf_info(test_file)
        print(f"Информация о PDF: {info}")
        
        # Извлечение текста
        result = processor.extract_text(test_file, max_pages=10)
        
        if result.get("success"):
            print(f"Успешно извлечен текст: {result['text_length']} символов")
            print(f"Количество страниц: {result['total_pages']}")
            print(f"Первые 500 символов: {result['text'][:500]}...")
        else:
            print(f"Ошибка: {result.get('error')}")
    else:
        print(f"Тестовый файл {test_file} не найден. Создайте PDF для тестирования.")