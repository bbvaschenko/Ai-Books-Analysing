"""
Telegram бот для взаимодействия с библиотекой
Только режим с системой агентов
"""
import os
import tempfile
import traceback
from typing import Dict, Any, List
import telebot
from telebot import types
from library_core import BookAnalyzer, BookData
import fitz  # PyMuPDF


class LibraryBot:
    """Класс Telegram бота для библиотеки с системой агентов"""

    def __init__(self, token: str):
        """Инициализация бота с системой агентов"""
        print("\n" + "="*60)
        print("🤖 ЗАПУСК БОТА С СИСТЕМОЙ АГЕНТОВ")
        print("="*60)

        self.bot = telebot.TeleBot(token)
        self.analyzer = BookAnalyzer(use_agents=True)  # Всегда с агентами
        self.user_states: Dict[int, Dict[str, Any]] = {}
        self.MAX_FILE_SIZE = 50 * 1024 * 1024

        # Счетчик поисков
        self.search_counts: Dict[int, int] = {}

        # Регистрация обработчиков
        self.register_handlers()

        print("✅ Бот инициализирован с системой из 5 агентов")
        print("="*60)

    def get_user_state(self, user_id: int) -> Dict[str, Any]:
        """Получение состояния пользователя"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {
                "is_download_message": False,
                "is_find_request": False,
                "last_message_id": None,
                "pending_file": None,
                "last_conversation_id": None,
                "search_count": 0
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
        if hasattr(file_info, 'mime_type') and file_info.mime_type:
            if file_info.mime_type == 'application/pdf':
                return True

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

    def extract_text_for_check(self, pdf_path: str) -> str:
        """Быстрое извлечение текста для проверки учебности"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i, page in enumerate(doc):
                if i >= 3:
                    break
                text += page.get_text()
            doc.close()
            return text[:2000]
        except Exception as e:
            print(f"Ошибка при быстром извлечении текста: {e}")
            return ""

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

            if state.get("last_message_id"):
                try:
                    self.bot.delete_message(chat_id=user_id, message_id=state["last_message_id"])
                except:
                    pass

            welcome_text = f"""
🤖 *Библиотека с системой интеллектуальных агентов*

Я помогу вам:
• 📥 Загружать PDF-учебники в библиотеку (до {self.format_file_size(self.MAX_FILE_SIZE)})
• 🔍 Находить нужную литературу через систему агентов
• ✅ Автоматически проверять учебные материалы

🤖 *Система агентов:*
- CoordinatorAgent: Управляет процессом
- SearchAgent: Ищет материалы  
- AnalysisAgent: Анализирует контент
- CriticAgent: Контролирует качество
- RecommendationAgent: Формирует ответы

Выберите действие:
            """

            sent_message = self.bot.send_message(
                message.chat.id,
                welcome_text,
                reply_markup=self.create_main_inline_keyboard(),
                parse_mode='Markdown'
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
                    text=f"📥 **Загрузка PDF-учебника**\n\nЗагрузите учебный PDF файл.\n\n*Требования:*\n- Формат: PDF\n- Макс. размер: {max_size_mb}MB\n- ✅ Файл будет проверен на учебную литературу\n\nПосле загрузки нажмите '✅ Подтвердить':",
                    reply_markup=self.create_confirmation_keyboard(),
                    parse_mode='Markdown'
                )

            elif call.data == 'find':
                state["is_find_request"] = True
                state["is_download_message"] = False

                self.bot.edit_message_text(
                    chat_id=user_id,
                    message_id=message_id,
                    text=f"🔍 **Поиск литературы через систему агентов**\n\nВведите ваш поисковый запрос:\n\n_Пример: учебник по математике для студентов, материалы по программированию..._",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )

            elif call.data == 'help':
                help_text = """
❓ **Помощь по использованию системы агентов**

🤖 *Как работает система:*
1. CoordinatorAgent получает ваш запрос
2. SearchAgent ищет материалы в базе
3. AnalysisAgent анализирует найденное
4. CriticAgent проверяет качество анализа
5. RecommendationAgent формирует ответ

📥 *Загрузка PDF:*
1. Нажмите "Загрузить PDF"
2. Отправьте PDF файл как документ
3. Нажмите "Подтвердить"
4. Система проверит и проанализирует файл

🔍 *Поиск литературы:*
1. Нажмите "Найти литературу"
2. Введите запрос
3. Система агентов найдет и проанализирует материалы
4. Вы получите объяснимые рекомендации
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
                    self.process_pending_file(user_id, state["pending_file"])
                    state["pending_file"] = None
                else:
                    self.bot.edit_message_text(
                        chat_id=user_id,
                        message_id=message_id,
                        text="⏳ **Ожидание файла**\n\nПожалуйста, отправьте PDF файл сейчас.",
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
                state["search_count"] = state.get("search_count", 0) + 1

                search_results = self.search_books(message.text, user_id)

                results_text = f"""
🔍 **Результаты поиска:** "{message.text}"

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

            if message.document.file_size > self.MAX_FILE_SIZE:
                max_size_formatted = self.format_file_size(self.MAX_FILE_SIZE)
                file_size_formatted = self.format_file_size(message.document.file_size)

                self.bot.reply_to(
                    message,
                    f"❌ **Файл слишком большой!**\n\n"
                    f"Размер файла: {file_size_formatted}\n"
                    f"Максимальный размер: {max_size_formatted}",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )
                state["is_download_message"] = False
                return

            if not self.is_valid_pdf(message.document, message.document.file_name):
                self.bot.reply_to(
                    message,
                    "❌ **Это не PDF файл!**\n\nПожалуйста, загрузите PDF документ.",
                    reply_markup=self.create_back_to_menu_keyboard(),
                    parse_mode='Markdown'
                )
                state["is_download_message"] = False
                return

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
                f"💾 *Размер:* {file_size_formatted}\n\n"
                f"Нажмите '✅ Подтвердить' для проверки и добавления в библиотеку.",
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
                    "❌ **Это фотография!**\n\nОтправьте PDF файл как *документ*.",
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
            self.bot.send_chat_action(user_id, 'upload_document')

            file_info_obj = self.bot.get_file(file_info['file_id'])
            file_download = self.bot.download_file(file_info_obj.file_path)

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_download)
                tmp_path = tmp_file.name

            self.bot.send_chat_action(user_id, 'typing')
            check_msg = self.bot.send_message(
                user_id,
                "🔍 *Проверяю файл на учебную литературу...*",
                parse_mode='Markdown'
            )

            quick_text = self.extract_text_for_check(tmp_path)

            if not quick_text or len(quick_text) < 100:
                try:
                    self.bot.delete_message(user_id, check_msg.message_id)
                except:
                    pass

                self.bot.send_message(
                    user_id,
                    "❌ **Не удалось извлечь текст для проверки.**\n\n"
                    "Убедитесь, что файл содержит текст, а не только изображения.",
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )

                try:
                    os.unlink(tmp_path)
                except:
                    pass

                return

            self.bot.edit_message_text(
                chat_id=user_id,
                message_id=check_msg.message_id,
                text="🔍 *Анализирую учебный материал...*",
                parse_mode='Markdown'
            )

            try:
                book_data = self.analyzer.analyze_book(tmp_path)

                if book_data is None:
                    try:
                        self.bot.delete_message(user_id, check_msg.message_id)
                    except:
                        pass

                    self.bot.send_message(
                        user_id,
                        "❌ **Файл не является учебной литературой!**\n\n"
                        "Загрузите учебник или учебное пособие.",
                        reply_markup=self.create_main_inline_keyboard(),
                        parse_mode='Markdown'
                    )

                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

                    return

                try:
                    self.bot.delete_message(user_id, check_msg.message_id)
                except:
                    pass

                report = self.format_book_report(book_data)

                self.bot.send_message(
                    user_id,
                    f"✅ **Файл успешно добавлен в библиотеку!**\n\n{report}",
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )

                try:
                    os.unlink(tmp_path)
                except:
                    pass

            except Exception as e:
                try:
                    self.bot.delete_message(user_id, check_msg.message_id)
                except:
                    pass

                error_msg = f"❌ **Ошибка при обработке файла:** {str(e)}"
                print(traceback.format_exc())
                self.bot.send_message(
                    user_id,
                    error_msg,
                    reply_markup=self.create_main_inline_keyboard(),
                    parse_mode='Markdown'
                )

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
        """Поиск книг через систему агентов"""
        state = self.get_user_state(user_id)

        # Используем систему агентов
        context = {
            "user_id": user_id,
            "user_level": self._detect_user_level(user_id),
            "search_count": state.get("search_count", 0),
            "preferred_format": "telegram"
        }

        try:
            result = self.analyzer.search_books_with_agents(query, context)

            if result.get("method") == "agent_system":
                agent_result = result.get("agent_system_result", {})
                recommendations = agent_result.get("recommendations", {})

                state["last_conversation_id"] = agent_result.get("conversation_id")

                return self._format_agent_search_results(query, recommendations, result.get("results", []))
            else:
                results = result.get("results", [])
                return self._format_search_results(query, results)

        except Exception as e:
            print(f"❌ Ошибка системы агентов: {e}")
            return "🔍 *Произошла ошибка при поиске. Попробуйте позже.*"

    def _detect_user_level(self, user_id: int) -> str:
        """Определение уровня пользователя"""
        state = self.get_user_state(user_id)
        history_count = state.get("search_count", 0)

        if history_count < 3:
            return "beginner"
        elif history_count < 10:
            return "intermediate"
        else:
            return "advanced"

    def _format_agent_search_results(self, query: str, recommendations: Dict, formatted_results: List) -> str:
        """Форматирование результатов системы агентов"""
        if not formatted_results:
            return "🔍 *По вашему запросу ничего не найдено.*"

        response = f"🤖 *Система агентов нашла для вас:*\n\n"
        response += f"*Запрос:* {query}\n"
        response += f"*Найдено рекомендаций:* {len(formatted_results)}\n\n"

        summary = recommendations.get("recommendation_summary", "")
        if summary:
            response += f"*📝 Резюме системы:*\n{summary[:300]}...\n\n"

        response += "*🏆 Топ рекомендации:*\n\n"

        for i, result in enumerate(formatted_results[:3], 1):
            response += f"{i}. *{result['filename']}*\n"
            response += f"   🆔 ID: {result['book_id']}\n"
            response += f"   🧭 Область: {result['area']}\n"

            explanation = result.get('explanation', '')
            if explanation:
                response += f"   💡 {explanation[:100]}...\n"

            score = result.get('score', 0)
            stars = int(score * 5) if isinstance(score, (int, float)) else 3
            response += f"   ⭐ Релевантность: {'★' * stars}\n\n"

        notes = recommendations.get("important_notes", {})
        if notes.get("limitations"):
            response += f"*⚠️ Ограничения:*\n"
            for limitation in notes["limitations"][:2]:
                response += f"• {limitation}\n"

        response += "*💡 Что хотите сделать дальше?*"

        return response

    def _format_search_results(self, query: str, results: List) -> str:
        """Форматирование результатов поиска"""
        if not results:
            return "🔍 *По вашему запросу ничего не найдено.*"

        response = f"*Найдено книг:* {len(results)}\n\n"

        for i, result in enumerate(results[:5], 1):
            response += f"{i}. *{result['filename']}*\n"
            response += f"   🆔 ID: {result['book_id']}\n"
            response += f"   🧭 Область: {result['area']}\n"
            if result.get('matching_tags'):
                response += f"   🔖 Теги: {', '.join(result['matching_tags'][:3])}\n"
            response += f"   ⭐ Релевантность: {'★' * result.get('score', 1)}\n\n"

        return response

    def start(self):
        """Запуск бота"""
        max_size_mb = self.MAX_FILE_SIZE // (1024 * 1024)

        print(f"🤖 Бот запущен с системой агентов!")
        print(f"📁 Максимальный размер файла: {max_size_mb}MB")
        print(f"🔍 Автономная проверка учебности: ВКЛЮЧЕНА")

        try:
            self.bot.polling(none_stop=True, interval=0, timeout=60)
        except Exception as e:
            print(f"❌ Ошибка бота: {e}")
            traceback.print_exc()