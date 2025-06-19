import asyncio
import logging
import os
import json
from typing import Dict, List, Optional
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
        logging.StreamHandler(),  # Вывод в консоль для Pella.app
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация
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

class RateLimiter:
    """Ограничитель частоты запросов"""
    def __init__(self):
        self.user_requests: Dict[int, List[datetime]] = {}
    
    def is_allowed(self, user_id: int) -> bool:
        """Проверка лимита запросов"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=Config.RATE_LIMIT_MINUTES)
        
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Очищаем старые запросы
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id] 
            if req_time > cutoff
        ]
        
        # Проверяем лимит
        if len(self.user_requests[user_id]) >= Config.RATE_LIMIT_REQUESTS:
            return False
        
        # Добавляем новый запрос
        self.user_requests[user_id].append(now)
        return True
    
    def cleanup_old_data(self):
        """Очистка старых данных для экономии памяти"""
        cutoff = datetime.now() - timedelta(hours=1)
        users_to_remove = []
        
        for user_id, requests in self.user_requests.items():
            self.user_requests[user_id] = [
                req_time for req_time in requests if req_time > cutoff
            ]
            if not self.user_requests[user_id]:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            del self.user_requests[user_id]

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
        users_to_remove = []
        
        for user_id, last_time in self.last_activity.items():
            if last_time < cutoff:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            self.clear_conversation(user_id)
        
        logger.info(f"🧹 Очищено {len(users_to_remove)} неактивных разговоров")

class GeminiBot:
    def __init__(self):
        # Настройка Gemini API
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Создание модели с настройками безопасности
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4000,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",  # или "gemini-1.5-pro-latest"
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter()
        self._cleanup_task = None
    
    async def start_cleanup_task(self):
        """Запуск задачи очистки после инициализации event loop"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("🧹 Задача периодической очистки запущена")
    
    async def _periodic_cleanup(self):
        """Периодическая очистка данных"""
        while True:
            await asyncio.sleep(3600)  # Каждый час
            try:
                self.conversation_manager.cleanup_inactive_conversations()
                self.rate_limiter.cleanup_old_data()
                logger.info("🧹 Выполнена периодическая очистка данных")
            except Exception as e:
                logger.error(f"Ошибка при очистке данных: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        # Запускаем cleanup task при первом использовании
        await self.start_cleanup_task()
        
        user_name = update.effective_user.first_name or "друг"
        welcome_message = f"""
🤖 Привет, {user_name}! Я бот с Google Gemini 2.0! ✨

📋 **Доступные команды:**
/start - Показать это сообщение
/clear - Очистить историю разговора
/help - Подробная справка
/status - Статус бота

💬 Просто напишите мне сообщение, и я отвечу с помощью Google Gemini!

⚡ Лимит: {Config.RATE_LIMIT_REQUESTS} сообщений в {Config.RATE_LIMIT_MINUTES} мин.
        """
        await update.message.reply_text(welcome_message)
        logger.info(f"👋 Новый пользователь: {update.effective_user.id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
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
• Максимум сообщений: {rate_limit} в {rate_minutes} мин.
• Максимальная длина сообщения: {max_length} символов
• История сохраняется в течение 24 часов

💡 **Советы:**
- Задавайте конкретные вопросы для лучших ответов
- Используйте /clear если нужно сменить тему
- Бот помнит контекст разговора
        """.format(
            rate_limit=Config.RATE_LIMIT_REQUESTS,
            rate_minutes=Config.RATE_LIMIT_MINUTES,
            max_length=Config.MAX_MESSAGE_LENGTH
        )
        await update.message.reply_text(help_text)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статус бота"""
        active_conversations = len(self.conversation_manager.conversations)
        user_id = update.effective_user.id
        user_messages = len(self.conversation_manager.get_conversation(user_id))
        
        status_text = f"""
📊 **Статус бота**

🤖 Модель: Google Gemini 2.0
🟢 Статус: Активен
💬 Активных разговоров: {active_conversations}
📝 Ваших сообщений в истории: {user_messages}

⚙️ **Конфигурация:**
• Лимит запросов: {Config.RATE_LIMIT_REQUESTS}/{Config.RATE_LIMIT_MINUTES}мин
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
        # Запускаем cleanup task при первом использовании
        await self.start_cleanup_task()
        
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Проверка лимита запросов
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text(
                f"⏰ Превышен лимит запросов! "
                f"Максимум {Config.RATE_LIMIT_REQUESTS} сообщений в {Config.RATE_LIMIT_MINUTES} минуту. "
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
        
        # Отправляем индикатор печати
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Добавляем сообщение пользователя в историю
            self.conversation_manager.add_message(user_id, "user", user_message)
            
            # Получаем историю разговора
            conversation_history = self.conversation_manager.get_conversation(user_id)
            
            # Отправляем запрос к Gemini
            if conversation_history:
                # Если есть история, используем chat
                chat = self.model.start_chat(history=conversation_history[:-1])  # Исключаем последнее сообщение
                response = await asyncio.to_thread(
                    chat.send_message, 
                    user_message
                )
            else:
                # Первое сообщение
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    user_message
                )
            
            # Получаем ответ
            gemini_response = response.text
            
            # Добавляем ответ в историю
            self.conversation_manager.add_message(user_id, "assistant", gemini_response)
            
            # Разбиваем длинные ответы на части
            if len(gemini_response) > Config.MAX_MESSAGE_LENGTH:
                chunks = [
                    gemini_response[i:i+Config.MAX_MESSAGE_LENGTH] 
                    for i in range(0, len(gemini_response), Config.MAX_MESSAGE_LENGTH)
                ]
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        await asyncio.sleep(0.5)  # Небольшая пауза между частями
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(gemini_response)
            
            logger.info(f"✅ Ответ отправлен пользователю {user_id}")
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "quota" in error_msg or "limit" in error_msg:
                await update.message.reply_text(
                    "⚠️ Превышен лимит запросов к Gemini API. "
                    "Попробуйте через несколько минут."
                )
                logger.warning(f"Gemini quota exceeded для пользователя {user_id}")
                
            elif "safety" in error_msg or "blocked" in error_msg:
                await update.message.reply_text(
                    "🛡️ Сообщение заблокировано фильтрами безопасности. "
                    "Попробуйте переформулировать вопрос."
                )
                logger.warning(f"Gemini safety filter для пользователя {user_id}")
                
            elif "api" in error_msg:
                await update.message.reply_text(
                    "❌ Ошибка Gemini API. Попробуйте позже или используйте /clear для сброса контекста."
                )
                logger.error(f"Gemini API error: {e}")
                
            else:
                await update.message.reply_text(
                    "❌ Произошла неожиданная ошибка. Попробуйте еще раз или используйте /clear."
                )
                logger.error(f"Unexpected error for user {user_id}: {e}")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глобальный обработчик ошибок"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.message:
            try:
                await update.message.reply_text(
                    "❌ Произошла ошибка. Попробуйте позже или используйте /clear для сброса."
                )
            except Exception:
                pass  # Игнорируем ошибки при отправке сообщения об ошибке

def create_application():
    """Создание приложения Telegram"""
    # Валидация конфигурации
    Config.validate()
    
    # Создаем экземпляр бота
    bot = GeminiBot()
    
    # Создаем приложение Telegram с оптимизированными настройками
    application = (Application.builder()
                  .token(Config.TELEGRAM_BOT_TOKEN)
                  .concurrent_updates(True)  # Параллельная обработка
                  .build())
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("status", bot.status_command))
    application.add_handler(CommandHandler("clear", bot.clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # Добавляем обработчик ошибок
    application.add_error_handler(bot.error_handler)
    
    return application

def main():
    """Главная функция запуска бота"""
    try:
        logger.info("🚀 Запуск Gemini Telegram Bot...")
        
        # Создаем приложение
        application = create_application()
        
        # Определяем режим запуска (для Pella.app используется polling)
        port = int(os.getenv("PORT", "8080"))
        
        if os.getenv("PELLA_APP") == "true":
            # Режим для Pella.app
            logger.info(f"🌐 Запуск в режиме Pella.app на порту {port}")
            application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                close_loop=False
            )
        else:
            # Локальный режим
            logger.info("💻 Запуск в локальном режиме")
            application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при запуске: {e}")
        raise

if __name__ == '__main__':
    main()