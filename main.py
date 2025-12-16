"""
Основной файл для запуска проекта
"""
import os
import sys
from telegram_bot import LibraryBot


def check_requirements():
    """Проверка необходимых директорий и файлов"""
    # Проверяем наличие директории с тегами
    if not os.path.exists("tags"):
        print("⚠️  Директория 'tags' не найдена!")
        print("Создайте директорию 'tags' и добавьте файлы с тегами:")
        print("  - разделы.txt")
        print("  - предметы.txt")
        print("  - классы.txt")
        print("  - авторы.txt")
        print("  - темы.txt")
        print("  - области_знаний.txt")
        return False

    # Проверяем наличие хотя бы одного файла тегов
    tag_files = [f for f in os.listdir("tags") if f.endswith(".txt")]
    if not tag_files:
        print("⚠️  В директории 'tags' нет файлов с тегами!")
        return False

    print(f"✅ Найдено файлов с тегами: {len(tag_files)}")

    # Создаем директорию для загруженных файлов
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
        print("✅ Создана директория 'uploads' для загруженных файлов")

    # Создаем директорию для логов
    if not os.path.exists("logs"):
        os.makedirs("logs")
        print("✅ Создана директория 'logs'")

    return True


def main():
    """Основная функция запуска"""
    print("=" * 50)
    print("🏫 Умная библиотека учебников")
    print("=" * 50)
    print("📁 Максимальный размер файла: 50MB")
    print("=" * 50)

    # Проверяем требования
    if not check_requirements():
        print("\n❌ Пожалуйста, настройте проект согласно инструкции выше.")
        sys.exit(1)

    # Токен бота
    BOT_TOKEN = '8299643533:AAFSCcKODXOm6eI7LT5FMMOFpJqXMfwikko'

    if not BOT_TOKEN or BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("\n❌ Укажите действительный токен Telegram бота в файле main.py")
        sys.exit(1)

    # Запускаем бота
    try:
        bot = LibraryBot(BOT_TOKEN)
        bot.start()
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


def analyze_example_pdf():
    """Функция для тестирования анализа PDF файла"""
    from library_core import BookAnalyzer

    print("\n🔬 Тестирование анализа PDF...")
    print("=" * 50)

    analyzer = BookAnalyzer()

    # Пример анализа файла
    test_pdf = "example.pdf"

    if os.path.exists(test_pdf):
        print(f"Анализирую файл: {test_pdf}")

        # Проверяем размер файла
        file_size = os.path.getsize(test_pdf)
        if file_size > 50 * 1024 * 1024:
            print(f"❌ Файл слишком большой: {file_size / (1024 * 1024):.1f}MB (максимум 50MB)")
            return

        book_data = analyzer.analyze_book(test_pdf)

        if book_data:
            print("\n✅ Анализ завершен успешно!")
            print(f"ID книги: {book_data.book_id}")
            print(f"Область знаний: {book_data.area}")
            print(f"Найденные теги:")

            for category, tags in book_data.tags.items():
                if tags:
                    print(f"  {category}: {', '.join(tags)}")

            # Сохраняем в базу
            analyzer.save_to_database(book_data)
            print(f"\n💾 Данные сохранены в analyzed_books.xlsx")
        else:
            print("❌ Не удалось проанализировать файл")
    else:
        print(f"❌ Тестовый файл не найден: {test_pdf}")
        print("\nСоздайте example.pdf или укажите путь к существующему PDF:")
        print("  python main.py --test /путь/к/файлу.pdf")


if __name__ == "__main__":
    # Можно запустить в двух режимах:
    # 1. Режим бота (по умолчанию)
    # 2. Режим тестирования анализа PDF

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        if len(sys.argv) > 2:
            # Переопределяем путь к тестовому файлу
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

            test_pdf = sys.argv[2]
            if os.path.exists(test_pdf):
                from library_core import BookAnalyzer

                print("\n🔬 Тестирование анализа PDF...")
                print(f"Файл: {test_pdf}")

                analyzer = BookAnalyzer()
                book_data = analyzer.analyze_book(test_pdf)

                if book_data:
                    print("\n✅ Анализ завершен успешно!")
                    analyzer.save_to_database(book_data)
                else:
                    print("❌ Не удалось проанализировать файл")
            else:
                print(f"❌ Файл не найден: {test_pdf}")
        else:
            analyze_example_pdf()
    else:
        main() 
