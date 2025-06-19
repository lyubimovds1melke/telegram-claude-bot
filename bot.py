import asyncio
import logging
import os
from typing import Dict, List
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования для облачного хостинга
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
    ]
)
logger = logging.getLogger(__name__)

# --- Конфигурация ---
class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MAX_CONVERSATION_LENGTH = int(os.getenv("MAX_CONVERSATION_LENGTH", "20"))
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "4000"))
    RATE_LIMIT_MINUTES = int(os.getenv("RATE_LIMIT_MINUTES", "1"))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "15"))

    @classmethod
    def validate(cls):
        """Валидация конфигурации"""
        if not cls.TELEGRAM_BOT_TOKEN or not cls.GEMINI_API_KEY:
            raise ValueError("❌ Отсутствуют обязательные переменные окружения: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY")
        logger.info("✅ Конфигурация валидна")

# --- Ограничитель частоты запросов ---
class RateLimiter:
    """Ограничитель частоты запросов"""
    def __init__(self):
        self.user_requests: Dict[int, List[datetime]] = {}

    def is_allowed(self, user_id: int) -> bool:
        """Проверка лимита запросов"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=Config.RATE_LIMIT_MINUTES)
        
        user_timestamps = self.user_requests.get(user_id, [])
        
        # Очищаем старые запросы
        valid_timestamps = [t for t in user_timestamps if t > cutoff]
        
        # Проверяем лимит
        if len(valid_timestamps) >= Config.RATE_LIMIT_REQUESTS:
            self.user_requests[user_id] = valid_timestamps
            return False
        
        # Добавляем новый запрос
        valid_timestamps.append(now)
        self.user_requests[user_id] = valid_timestamps
        return True

    def cleanup_old_data(self):
        """Очистка старых данных для экономии памяти"""
        cutoff = datetime.now() - timedelta(hours=1)
        users_to_remove = []
        
        for user_id, requests in self.user_requests.items():
            valid_requests = [req_time for req_time in requests if req_time > cutoff]
            if not valid_requests:
                users_to_remove.append(user_id)
            else:
                self.user_requests[user_id] = valid_requests
        
        for user_id in users_to_remove:
            del self.user_requests[user_id]
        logger.info(f"🧹 RateLimiter: Очищено данных для {len(users_to_remove)} пользователей.")

# --- Менеджер разговоров ---
class ConversationManager:
    """Менеджер разговоров с оптимизацией памяти для Gemini"""
    def __init__(self):
        self.conversations: Dict[int, List[Dict]] = {}
        self.last_activity: Dict[int, datetime] = {}

    def add_message(self, user_id: int, role: str, content: str):
        """Добавление сообщения в разговор (Gemini format)"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        # Gemini использует 'user' и 'model' роли
        gemini_role = "user" if role == "user" else "model"
        
        self.conversations[user_id].append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })
        
        self.last_activity[user_id] = datetime.now()
        
        # Ограничиваем длину разговора
        if len(self.conversations[user_id]) > Config.MAX_CONVERSATION_LENGTH:
            self.conversations[user_id] = self.conversations[user_id][-Config.MAX_CONVERSATION_LENGTH:]

    def get_conversation(self, user_id: int) -> List[Dict]:
        """Получение разговора пользователя в формате Gemini"""
        return self.conversations.get(user_id, [])

    def clear_conversation(self, user_id: int):
        """Очистка разговора пользователя"""
        if user_id in self.conversations:
            del self.conversations[user_id]
        if user_id in self.last_activity:
            del self.last_activity[user_id]

    def cleanup_inactive_conversations(self):
        """Очистка неактивных разговоров"""
        cutoff = datetime.now() - timedelta(hours=24)
        users_to_remove = [
            user_id for user_id, last_time in self.last_activity.items() if last_time < cutoff
        ]
        
        for user_id in users_to_remove:
            self.clear_conversation(user_id)
        
        if users_to_remove:
            logger.info(f"🧹 ConversationManager: Очищено {len(users_to_remove)} неактивных разговоров.")

# --- Основной класс бота ---
class GeminiBot:
    def __init__(self):
        # Настройка Gemini API
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096, # Увеличил для соответствия модели
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Использование более стабильной и распространенной модели
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter()

    async def post_init(self, application: Application):
        """Запускает фоновую задачу после инициализации приложения."""
        asyncio.create_task(self._periodic_cleanup())
        logger.info("✅ Бот инициализирован, фоновая задача очистки запущена.")
    
    async def _periodic_cleanup(self):
        """Периодическая очистка данных в фоне."""
        while True:
            await asyncio.sleep(3600)  # Каждый час
            try:
                self.conversation_manager.cleanup_inactive_conversations()
                self.rate_limiter.cleanup_old_data()
                logger.info("🧹 Выполнена периодическая очистка данных.")
            except Exception as e:
                logger.error(f"❌ Ошибка при периодической очистке: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user_name = update.effective_user.first_name or "друг"
        welcome_message = f"""
🤖 Привет, {user_name}! Я бот с Google Gemini! ✨

📋 **Доступные команды:**
/start - Показать это сообщение
/clear - Очистить историю разговора
/help - Подробная справка
/status - Статус бота

💬 Просто напишите мне сообщение, и я отвечу с помощью Google Gemini!

⚡ Лимит: {Config.RATE_LIMIT_REQUESTS} сообщений в {Config.RATE_LIMIT_MINUTES} мин.
        """
        await update.message.reply_text(welcome_message)
        logger.info(f"👋 Новый пользователь: {update.effective_user.id} ({update.effective_user.username})")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = f"""
🔧 **Подробная справка**

**Команды:**
/start - Начать работу с ботом
/clear - Очистить историю разговора
/help - Показать эту справку
/status - Статус и статистика бота

**Возможности:**
• Умные ответы на любые вопросы
• Помощь с программированием
• Объяснение сложных концепций
• Творческие задачи
• Поддержка русского и английского языков

**Ограничения:**
• Максимум сообщений: {Config.RATE_LIMIT_REQUESTS} в {Config.RATE_LIMIT_MINUTES} мин.
• Максимальная длина сообщения: {Config.MAX_MESSAGE_LENGTH} символов
• История сохраняется в течение 24 часов

💡 **Советы:**
- Задавайте конкретные вопросы для лучших ответов
- Используйте /clear если нужно сменить тему
- Бот помнит контекст разговора (последние {Config.MAX_CONVERSATION_LENGTH} сообщений)
        """
        await update.message.reply_text(help_text)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статус бота"""
        active_conversations = len(self.conversation_manager.conversations)
        user_id = update.effective_user.id
        user_messages = len(self.conversation_manager.get_conversation(user_id))
        
        status_text = f"""
📊 **Статус бота**

🤖 Модель: {self.model.model_name}
🟢 Статус: Активен
💬 Активных разговоров: {active_conversations}
📝 Ваших сообщений в истории: {user_messages}

⚙️ **Конфигурация:**
• Лимит запросов: {Config.RATE_LIMIT_REQUESTS}/{Config.RATE_LIMIT_MINUTES} мин
• Макс. длина сообщения: {Config.MAX_MESSAGE_LENGTH}
• Макс. история: {Config.MAX_CONVERSATION_LENGTH} сообщений
        """
        await update.message.reply_text(status_text)
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистка истории разговора"""
        user_id = update.effective_user.id
        self.conversation_manager.clear_conversation(user_id)
        await update.message.reply_text("✅ История разговора очищена! Можете начать с чистого листа.")
        logger.info(f"🗑️ Пользователь {user_id} очистил историю")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка обычных сообщений"""
        user_id = update.effective_user.id
        user_message = update.message.text
        
        if not user_message:
            return

        # Проверка лимита запросов
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text(
                f"⏰ Превышен лимит запросов! "
                f"Максимум {Config.RATE_LIMIT_REQUESTS} сообщений за {Config.RATE_LIMIT_MINUTES} мин. "
                f"Попробуйте чуть позже."
            )
            return
        
        # Проверка длины сообщения
        if len(user_message) > Config.MAX_MESSAGE_LENGTH:
            await update.message.reply_text(
                f"📝 Сообщение слишком длинное! "
                f"Максимум {Config.MAX_MESSAGE_LENGTH} символов. "
                f"Ваше: {len(user_message)} символов."
            )
            return
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Добавляем сообщение пользователя в историю
            self.conversation_manager.add_message(user_id, "user", user_message)
            conversation_history = self.conversation_manager.get_conversation(user_id)
            
            # Упрощенный и более надежный вызов API
            response = await asyncio.to_thread(
                self.model.generate_content,
                conversation_history
            )
            
            gemini_response = response.text
            
            # Добавляем ответ в историю
            self.conversation_manager.add_message(user_id, "model", gemini_response)
            
            # Разбиваем длинные ответы на части
            if len(gemini_response) > Config.MAX_MESSAGE_LENGTH:
                for i in range(0, len(gemini_response), Config.MAX_MESSAGE_LENGTH):
                    chunk = gemini_response[i:i+Config.MAX_MESSAGE_LENGTH]
                    await update.message.reply_text(chunk)
                    await asyncio.sleep(0.5)
            else:
                await update.message.reply_text(gemini_response)
            
            logger.info(f"✅ Ответ отправлен пользователю {user_id}")
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"❌ Ошибка при обработке сообщения для {user_id}: {e}", exc_info=True)
            
            if "quota" in error_msg or "limit" in error_msg:
                reply = "⚠️ Превышен лимит запросов к Gemini API. Попробуйте через несколько минут."
            elif "safety" in error_msg or "blocked" in error_msg:
                reply = "🛡️ Ваш запрос или ответ были заблокированы фильтрами безопасности. Попробуйте переформулировать."
            elif "api" in error_msg or "key" in error_msg:
                reply = "❌ Ошибка конфигурации Gemini API. Возможно, неверный ключ."
            else:
                reply = "❌ Произошла неожиданная ошибка. Попробуйте еще раз или используйте /clear для сброса контекста."
            
            await update.message.reply_text(reply)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Глобальный обработчик ошибок"""
        logger.error(f"Исключение при обработке апдейта {update}:", exc_info=context.error)
        
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ Произошла внутренняя ошибка. Попробуйте позже или используйте /clear для сброса."
                )
            except Exception as e:
                logger.error(f"Не удалось отправить сообщение об ошибке пользователю: {e}")

# --- Запуск бота ---
def main():
    """Главная функция запуска бота"""
    try:
        logger.info("🚀 Запуск Gemini Telegram Bot...")
        
        Config.validate()
        bot = GeminiBot()
        
        application = (Application.builder()
                       .token(Config.TELEGRAM_BOT_TOKEN)
                       .concurrent_updates(True)
                       .post_init(bot.post_init) # Правильный запуск фоновых задач
                       .build())
        
        application.add_handler(CommandHandler("start", bot.start_command))
        application.add_handler(CommandHandler("help", bot.help_command))
        application.add_handler(CommandHandler("status", bot.status_command))
        application.add_handler(CommandHandler("clear", bot.clear_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
        
        application.add_error_handler(bot.error_handler)
        
        logger.info("🤖 Бот готов к работе. Запуск polling...")
        # Этот метод подходит и для локального запуска, и для многих облачных сервисов,
        # которые не требуют веб-сервера (например, Pella.app).
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
            
    except (ValueError, KeyError) as e:
        logger.critical(f"❌ Критическая ошибка конфигурации: {e}")
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка при запуске бота: {e}", exc_info=True)

if __name__ == '__main__':
    main()