"""
Telegram бот для взаимодействия с библиотекой
"""
import os
import tempfile
import traceback
from typing import Dict, Any
import telebot
from telebot import types
from library_core import BookAnalyzer, BookData


class LibraryBot:
    """Класс Telegram бота для библиотеки"""

    def __init__(self, token: str):
        """Инициализация бота"""
        self.bot = telebot.TeleBot(token)
        self.analyzer = BookAnalyzer()
        self.user_states: Dict[int, Dict[str, Any]] = {}
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB в байтах

        # Регистрация обработчиков
        self.register_handlers()

    def get_user_state(self, user_id: int) -> Dict[str, Any]:
        """Получение состояния пользователя"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "is_download_message": False,
                "is_find_request": False,
                "last_message_id": None,
                "pending_file": None
            }
        return self.user_states[user_id]

    def create_main_inline_keyboard(self):
        """Создает основную инлайн-клавиатуру"""
        keyboard = types.InlineKeyboardMarkup(row_width=2)

        btn_download = types.InlineKeyboardButton('📥 Загрузить PDF', callback_data='download')
        btn_find = types.InlineKeyboardButton('🔍 Найти литературу', callback_data='find')
        btn_help = types.InlineKeyboardButton('❓ Помощь', callback_data='help')
        btn_support = types.InlineKeyboardButton('👥 Поддержка', callback_data='support')

        keyboard.add(btn_download, btn_find, btn_help, btn_support)
        return keyboard

    def create_back_to_menu_keyboard(self):
        """Клавиатура с кнопкой возврата в меню"""
        keyboard = types.InlineKeyboardMarkup()
        btn_back = types.InlineKeyboardButton('⬅️ Назад в меню', callback_data='back_to_menu')
        keyboard.add(btn_back)
        return keyboard

    def create_confirmation_keyboard(self):
        """Клавиатура для подтверждения/отмены"""
        keyboard = types.InlineKeyboardMarkup(row_width=2)
        btn_confirm = types.InlineKeyboardButton('✅ Подтвердить', callback_data='confirm_upload')
        btn_cancel = types.InlineKeyboardButton('❌ Отмена', callback_data='back_to_menu')
        keyboard.add(btn_confirm, btn_cancel)
        return keyboard

    def is_valid_pdf(self, file_info, file_name: str) -> bool:
        """Проверяет, является ли файл PDF"""
        # Проверка по MIME типу
        if hasattr(file_info, 'mime_type') and file_info.mime_type:
            if file_info.mime_type == 'application/pdf':
                return True

        # Проверка по расширению файла
        if file_name:
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in ['.pdf']:
                return True

        return False

    def format_file_size(self, size_in_bytes: int) -> str:
        """Форматирует размер файла в читаемый вид"""
        if size_in_bytes < 1024:
            return f"{size_in_bytes} B"
        elif size_in_bytes < 1024 * 1024:
            return f"{size_in_bytes / 1024:.1f} KB"
        elif size_in_bytes < 1024 * 1024 * 1024:
            return f"{size_in_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"

    def register_handlers(self):
        """Регистрация всех обработчиков бота"""

        @self.bot.message_handler(func=lambda message:
            message.text and
            message.text.lower().replace('/', '') in ['start', 'старт', 'начать']
        )
        def handle_start(message):
            user_id = message.chat.id
            state = self.get_user_state(user_id)
            state["is_download_message"] = False
            state["is_find_request"] = False
            state["pending_file"] = None

            # Удаляем предыдущее сообщение с меню если есть
            if state.get("last_message_id"):
                try:
                    self.bot.delete_message(chat_id=user_id, message_id=state["last_message_id"])
                except:
                    pass

            welcome_text = f"""
🤖 Добро пожаловать в Библиотечного Бота!

Я помогу вам:
• 📥 Загружать PDF-файлы в библиотеку (до {self.format_file_size(self.MAX_FILE_SIZE)})
• 🔍 Находить нужную литературу по запросу

Выберите действие:
            """

            sent_message = self.bot.send_message(
                message.chat.id,
                welcome_text,
                reply_markup=self.create_main_inline_keyboard()
            )
            state["last_message_id"] = sent_message.message_id

        @self.bot.callback_query_handler(func=lambda call: True)
        def handle_callback(call):
            user_id = call.message.chat.id
            message_id = call.message.message_id
            state = self.get_user_state(user_id)
            state["last_message_id"] = message_id

            if call.data == 'download':
                state["is_download_message"] = True
                state["is_find_request"] = False

                max_size_mb = self.MAX_FILE_SIZE // (1024 * 1024)

                self.bot.edit_message_text(
                    chat_id=user_id,
                    message_id=message_id,
                    text=f"📥 **Загрузка PDF-файла**\n\nПожалуйста, загрузите документ в формате PDF.\n\n*Требования:*\n- Формат файла: PDF\n- Максимальный размер: {max_size_mb}MB ({self.format_file_size(self.MAX_FILE_SIZE)})\n\nПосле загрузки файла нажмите кнопку ниже:",
                    reply_markup=self.create_confirmation_keyboard(),
                    parse_mode='Markdown'
                )

            elif call.data == 'find':
                state["is_find_request"] = True
                state["is_download_message"] = False

                self.bot.edit_message_text(
                    chat_id=user_id,
                    message_id=message_id,
                    text="🔍 **Поиск литературы**\n\nВведите ваш поисковый запрос:\n\n_Например: математика, программирование, физика..._",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )

            elif call.data == 'help':
                max_size_mb = self.MAX_FILE_SIZE // (1024 * 1024)

                help_text = f"""
❓ **Помощь**

📥 **Загрузить PDF** - добавление документа в библиотеку (до {max_size_mb}MB)
🔍 **Найти литературу** - поиск по базе документов
👥 **Поддержка** - связь с технической поддержкой

*Как загрузить PDF:*
1. Нажмите "Загрузить PDF"
2. Отправьте PDF файл как документ (не фото!)
3. Нажмите "Подтвердить"

Просто нажмите на нужную кнопку ниже 👇
                """

                self.bot.edit_message_text(
                    chat_id=user_id,
                    message_id=message_id,
                    text=help_text,
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )

            elif call.data == 'support':
                support_text = """
👥 **Поддержка**

Если у вас возникли проблемы или вопросы:

📧 Email: internationsupport@gmail.com
💬 Чат: @internationsupport
                """

                self.bot.edit_message_text(
                    chat_id=user_id,
                    message_id=message_id,
                    text=support_text,
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )

            elif call.data == 'back_to_menu':
                handle_start(call.message)

            elif call.data == 'confirm_upload':
                if state.get("pending_file"):
                    # Обрабатываем ожидающий файл
                    self.process_pending_file(user_id, state["pending_file"])
                    state["pending_file"] = None
                else:
                    self.bot.edit_message_text(
                        chat_id=user_id,
                        message_id=message_id,
                        text="⏳ **Ожидание файла**\n\nПожалуйста, отправьте PDF файл прямо сейчас.",
                        reply_markup=self.create_back_to_menu_keyboard(),
                        parse_mode='Markdown'
                    )

            self.bot.answer_callback_query(call.id)

        @self.bot.message_handler(content_types=['text'])
        def handle_text(message):
            user_id = message.chat.id
            state = self.get_user_state(user_id)

            if state["is_find_request"]:
                self.bot.send_chat_action(user_id, 'typing')
                search_results = self.search_books(message.text, user_id)

                results_text = f"""
🔍 **Результаты поиска по запросу:** "{message.text}"

{search_results}

Что хотите сделать дальше?
                """

                self.bot.send_message(
                    user_id,
                    results_text,
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )

                state["is_find_request"] = False

            else:
                handle_start(message)

        @self.bot.message_handler(content_types=['document'])
        def handle_document(message):
            user_id = message.chat.id
            state = self.get_user_state(user_id)

            if not state["is_download_message"]:
                self.bot.reply_to(
                    message,
                    "⚠️ Сначала выберите 'Загрузить PDF' из меню",
                    reply_markup=self.create_main_inline_keyboard()
                )
                return

            # Проверка размера файла
            if message.document.file_size > self.MAX_FILE_SIZE:
                max_size_formatted = self.format_file_size(self.MAX_FILE_SIZE)
                file_size_formatted = self.format_file_size(message.document.file_size)

                self.bot.reply_to(
                    message,
                    f"❌ **Файл слишком большой!**\n\n"
                    f"Размер файла: {file_size_formatted}\n"
                    f"Максимальный размер: {max_size_formatted}\n\n"
                    f"Пожалуйста, загрузите файл меньшего размера.",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )
                state["is_download_message"] = False
                return

            # Проверка формата файла
            if not self.is_valid_pdf(message.document, message.document.file_name):
                self.bot.reply_to(
                    message,
                    "❌ **Это не PDF файл!**\n\nПожалуйста, загрузите документ в формате PDF.",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )
                state["is_download_message"] = False
                return

            # Сохраняем информацию о файле для подтверждения
            file_size_formatted = self.format_file_size(message.document.file_size)
            state["pending_file"] = {
                'file_id': message.document.file_id,
                'file_name': message.document.file_name or f"document_{message.document.file_id}.pdf",
                'file_size': message.document.file_size,
                'file_size_formatted': file_size_formatted
            }

            self.bot.reply_to(
                message,
                f"✅ **Файл получен!**\n\n"
                f"📄 *Название:* {state['pending_file']['file_name']}\n"
                f"💾 *Размер:* {file_size_formatted}\n"
                f"👤 *Отправил:* {message.from_user.first_name}\n\n"
                f"Нажмите '✅ Подтвердить' для добавления в библиотеку.",
                reply_markup=self.create_confirmation_keyboard(),
                parse_mode='Markdown'
            )

        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message):
            user_id = message.chat.id
            state = self.get_user_state(user_id)

            if state["is_download_message"]:
                self.bot.reply_to(
                    message,
                    "❌ **Это фотография, а не PDF файл!**\n\nПожалуйста отправьте PDF файл как *документ*.",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )
                state["is_download_message"] = False
            else:
                self.bot.reply_to(
                    message,
                    "⚠️ Я работаю только с PDF документами.",
                    reply_markup=self.create_main_inline_keyboard()
                )

    def process_pending_file(self, user_id: int, file_info: Dict):
        """Обработка загруженного файла"""
        try:
            # Показываем статус обработки
            self.bot.send_chat_action(user_id, 'upload_document')

            # Скачиваем файл
            file_info_obj = self.bot.get_file(file_info['file_id'])
            file_download = self.bot.download_file(file_info_obj.file_path)

            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_download)
                tmp_path = tmp_file.name

            # Показываем статус анализа
            processing_msg = self.bot.send_message(
                user_id,
                f"⏳ **Обработка файла...**\n\n"
                f"📄 *Файл:* {file_info['file_name']}\n"
                f"💾 *Размер:* {file_info['file_size_formatted']}\n\n"
                f"Извлекаю текст и анализирую содержимое...",
                parse_mode='Markdown'
            )

            # Анализируем книгу
            self.bot.send_chat_action(user_id, 'typing')
            book_data = self.analyzer.analyze_book(tmp_path)

            if book_data:
                # Сохраняем в базу
                self.analyzer.save_to_database(book_data)

                # Удаляем сообщение о процессе
                try:
                    self.bot.delete_message(user_id, processing_msg.message_id)
                except:
                    pass

                # Формируем отчет
                report = self.format_book_report(book_data)

                self.bot.send_message(
                    user_id,
                    f"✅ **Файл успешно добавлен в библиотеку!**\n\n{report}",
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )
            else:
                try:
                    self.bot.delete_message(user_id, processing_msg.message_id)
                except:
                    pass

                self.bot.send_message(
                    user_id,
                    "❌ **Не удалось проанализировать файл.**\n\nУбедитесь, что это учебный PDF с текстом.",
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )

            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except:
                pass

        except Exception as e:
            error_msg = f"❌ **Ошибка при обработке файла:** {str(e)}"
            print(traceback.format_exc())
            self.bot.send_message(
                user_id,
                error_msg,
                reply_markup=self.create_main_inline_keyboard(),
                parse_mode='Markdown'
            )

    def format_book_report(self, book_data: BookData) -> str:
        """Форматирование отчета о книге"""
        report = f"""
📚 *ID книги:* {book_data.book_id}
📄 *Название:* {book_data.filename}
🧭 *Область знаний:* {book_data.area}

*Найденные теги:*
"""

        for category, tags in book_data.tags.items():
            if tags:
                report += f"• *{category.capitalize()}:* {', '.join(tags)}\n"

        return report

    def search_books(self, query: str, user_id: int) -> str:
        """Поиск книг по запросу"""
        results = self.analyzer.search_books(query)

        if not results:
            return "🔍 *По вашему запросу ничего не найдено.*\n\nПопробуйте изменить запрос или использовать другие ключевые слова."

        response = f"*Найдено книг:* {len(results)}\n\n"

        for i, result in enumerate(results, 1):
            response += f"{i}. *{result['filename']}*\n"
            response += f"   🆔 ID: {result['book_id']}\n"
            response += f"   🧭 Область: {result['area']}\n"
            if result['matching_tags']:
                response += f"   🔖 Совпадающие теги: {', '.join(result['matching_tags'][:3])}\n"
            response += f"   ⭐ Релевантность: {'★' * result['score']}\n\n"

        return response

    def start(self):
        """Запуск бота"""
        max_size_mb = self.MAX_FILE_SIZE // (1024 * 1024)
        print(f"🤖 Бот запущен и готов к работе!")
        print(f"📁 Максимальный размер файла: {max_size_mb}MB")

        try:
            self.bot.polling(none_stop=True, interval=0, timeout=60)
        except Exception as e:
            print(f"❌ Ошибка бота: {e}")
            traceback.print_exc()