import asyncio
import logging
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
from telegram import Update, Message
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, SafetySetting, HarmCategory, Part
from dotenv import load_dotenv

# --- Начальная настройка ---
load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Конфигурация ---
class Config:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Системная инструкция для задания роли и поведения модели
    SYSTEM_INSTRUCTION = os.getenv(
        "SYSTEM_INSTRUCTION",
        "Ты — дружелюбный и полезный ассистент Gemini в Telegram. Отвечай на русском языке."
    )
    # Используем ОГРОМНЫЙ контекст Gemini 1.5 Pro, но с разумным лимитом
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "700000")) # 700k токенов
    MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "100")) # Лимит на число сообщений
    MAX_MESSAGE_LENGTH = 4096 # Лимит Telegram
    RATE_LIMIT_MINUTES = int(os.getenv("RATE_LIMIT_MINUTES", "1"))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20")) # 1.5 Pro позволяет больше запросов

    @classmethod
    def validate(cls):
        if not cls.TELEGRAM_BOT_TOKEN or not cls.GEMINI_API_KEY:
            raise ValueError("❌ Отсутствуют обязательные переменные окружения: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY")
        logger.info("✅ Конфигурация валидна")

# --- Управление данными ---
# Классы RateLimiter и ConversationManager остаются почти без изменений,
# так как их логика не зависит от модели.
# Мы лишь добавим поддержку хранения 'parts' вместо 'content'.
class RateLimiter: # ... (без изменений, можно скопировать из вашего кода) ...
    def __init__(self): self.user_requests: Dict[int, List[datetime]] = {}
    def is_allowed(self, user_id: int) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(minutes=Config.RATE_LIMIT_MINUTES)
        timestamps = self.user_requests.get(user_id, [])
        valid_timestamps = [t for t in timestamps if t > cutoff]
        if len(valid_timestamps) >= Config.RATE_LIMIT_REQUESTS:
            self.user_requests[user_id] = valid_timestamps
            return False
        valid_timestamps.append(now)
        self.user_requests[user_id] = valid_timestamps
        return True
    def cleanup_old_data(self):
        cutoff = datetime.now() - timedelta(hours=1)
        to_remove = [uid for uid, ts in self.user_requests.items() if not any(t > cutoff for t in ts)]
        for uid in to_remove: del self.user_requests[uid]
        if to_remove: logger.info(f"🧹 RateLimiter: очищены данные для {len(to_remove)} пользователей.")

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[int, List[Dict[str, Any]]] = {}
        self.last_activity: Dict[int, datetime] = {}
    def add_message(self, user_id: int, role: str, parts: List[Part]):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append({"role": role, "parts": parts})
        self.last_activity[user_id] = datetime.now()
        # Ограничиваем по количеству сообщений как запасной механизм
        if len(self.conversations[user_id]) > Config.MAX_CONVERSATION_MESSAGES:
            self.conversations[user_id] = self.conversations[user_id][-Config.MAX_CONVERSATION_MESSAGES:]
    def get_conversation(self, user_id: int) -> List[Dict[str, Any]]:
        return self.conversations.get(user_id, [])
    def clear_conversation(self, user_id: int):
        if user_id in self.conversations: del self.conversations[user_id]
        if user_id in self.last_activity: del self.last_activity[user_id]
    def cleanup_inactive_conversations(self):
        cutoff = datetime.now() - timedelta(hours=24)
        to_remove = [uid for uid, t in self.last_activity.items() if t < cutoff]
        for uid in to_remove: self.clear_conversation(uid)
        if to_remove: logger.info(f"🧹 ConversationManager: очищено {len(to_remove)} неактивных разговоров.")

# --- Основной класс бота ---
class GeminiBot:
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # 1. Смена модели на 1.5 Pro и использование системной инструкции
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            system_instruction=Config.SYSTEM_INSTRUCTION,
            generation_config=GenerationConfig(
                temperature=0.75,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            ),
            safety_settings=[
                SafetySetting(HarmCategory.HARM_CATEGORY_HARASSMENT, "BLOCK_MEDIUM_AND_ABOVE"),
                SafetySetting(HarmCategory.HARM_CATEGORY_HATE_SPEECH, "BLOCK_MEDIUM_AND_ABOVE"),
                SafetySetting(HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "BLOCK_MEDIUM_AND_ABOVE"),
                SafetySetting(HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "BLOCK_MEDIUM_AND_ABOVE"),
            ]
        )
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter()

    # ... post_init и _periodic_cleanup остаются без изменений ...
    async def post_init(self, application: Application): asyncio.create_task(self._periodic_cleanup())
    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(3600)
            try:
                self.conversation_manager.cleanup_inactive_conversations()
                self.rate_limiter.cleanup_old_data()
            except Exception as e: logger.error(f"❌ Ошибка при периодической очистке: {e}")

    # --- Команды ---
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_name = update.effective_user.first_name or "друг"
        # Обновленное приветствие с упоминанием 1.5 Pro
        await update.message.reply_text(
            f"👋 Привет, {user_name}! Я бот на базе **Google Gemini 1.5 Pro**.\n\n"
            "Я помню большой контекст нашего разговора и даже могу анализировать изображения! "
            "Просто отправь мне картинку с подписью.\n\n"
            "Используй /clear, чтобы начать разговор с чистого листа.",
            parse_mode=ParseMode.MARKDOWN
        )

    # ... help_command, status_command, clear_command можно оставить или адаптировать ...
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.conversation_manager.clear_conversation(update.effective_user.id)
        await update.message.reply_text("✅ История разговора очищена!")

    # 2. Оптимизация контекста по токенам, а не по сообщениям
    async def _prune_history_by_tokens(self, user_id: int) -> List[Dict[str, Any]]:
        """Обрезает историю разговора, чтобы она не превышала лимит токенов."""
        history = self.conversation_manager.get_conversation(user_id)
        while True:
            # Считаем токены асинхронно
            token_count = await self.model.count_tokens_async(history)
            if token_count.total_tokens <= Config.MAX_CONTEXT_TOKENS:
                break
            # Если превышен лимит, удаляем самые старые сообщения (кроме первого)
            if len(history) > 1:
                history.pop(0)
            else: # Если даже одно сообщение слишком большое
                break 
        
        self.conversation_manager.conversations[user_id] = history
        return history

    # 3. Единый обработчик для текста и изображений
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        message = update.message
        
        # Проверка лимита запросов
        if not self.rate_limiter.is_allowed(user_id):
            await message.reply_text("⏰ Превышен лимит запросов. Попробуйте чуть позже.")
            return

        parts = []
        # Собираем части сообщения (текст и/или фото)
        if message.text:
            parts.append(Part.from_text(message.text))
        if message.photo:
            photo_file = await message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            parts.append(Part.from_data(bytes(photo_bytes), mime_type="image/jpeg"))
            # Добавляем подпись к фото как текстовую часть, если она есть
            if message.caption:
                parts.append(Part.from_text(message.caption))

        if not parts:
            return

        try:
            await context.bot.send_chat_action(chat_id=user_id, action="typing")
            # Добавляем сообщение пользователя в историю
            self.conversation_manager.add_message(user_id, "user", parts)
            
            # Обрезаем историю по токенам перед отправкой
            pruned_history = await self._prune_history_by_tokens(user_id)
            
            # 4. Стриминг ответа для лучшего UX
            placeholder_message = await message.reply_text("🧠 Думаю...")
            
            # Используем нативный асинхронный вызов со стримингом
            response_stream = await self.model.generate_content_async(
                pruned_history,
                stream=True
            )
            
            await self._stream_and_edit_message(response_stream, placeholder_message)

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке сообщения для {user_id}: {e}", exc_info=True)
            await message.reply_text("❌ Произошла ошибка. Попробуйте снова или /clear.")
    
    # 5. Метод для стриминга и редактирования сообщения
    async def _stream_and_edit_message(self, stream, message: Message):
        """Читает стрим от Gemini и плавно редактирует сообщение в Telegram."""
        full_response = ""
        buffer = ""
        last_edit_time = time.time()
        
        async for chunk in stream:
            # Иногда в стриме могут быть пустые чанки или чанки без текста
            if text_part := chunk.text:
                buffer += text_part
                full_response += text_part
            
            current_time = time.time()
            # Редактируем сообщение раз в 1.2 секунды или если накопилось много текста
            if (current_time - last_edit_time > 1.2 and buffer) or len(buffer) > 200:
                try:
                    # Добавляем "█" для имитации курсора
                    await message.edit_text(full_response + "█", parse_mode=ParseMode.MARKDOWN)
                except BadRequest as e:
                    # Игнорируем ошибку, если текст не изменился
                    if "Message is not modified" not in str(e):
                        logger.warning(f"Ошибка при редактировании сообщения: {e}")
                
                last_edit_time = current_time
                buffer = ""
        
        # Финальное редактирование, чтобы убрать курсор и отправить полный текст
        if full_response:
            try:
                await message.edit_text(full_response, parse_mode=ParseMode.MARKDOWN)
            except BadRequest as e:
                if "Message is not modified" not in str(e):
                    logger.warning(f"Ошибка при финальном редактировании: {e}")
        else:
            # Если модель не вернула текст (например, из-за safety settings)
            await message.edit_text("Ответ не был получен.", parse_mode=ParseMode.MARKDOWN)
        
        # Добавляем ответ модели в историю
        self.conversation_manager.add_message(message.chat.id, "model", [Part.from_text(full_response)])

# --- Запуск бота ---
def main():
    try:
        logger.info("🚀 Запуск Gemini 1.5 Pro Telegram Bot...")
        Config.validate()
        bot = GeminiBot()
        
        application = (Application.builder()
                       .token(Config.TELEGRAM_BOT_TOKEN)
                       .concurrent_updates(True)
                       .post_init(bot.post_init)
                       .build())
        
        # Добавляем обработчик для фото И текста в одном месте
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.PHOTO) & ~filters.COMMAND,
            bot.handle_message
        ))
        application.add_handler(CommandHandler("start", bot.start_command))
        application.add_handler(CommandHandler("clear", bot.clear_command))
        # application.add_error_handler(...) # Ваш обработчик ошибок
        
        logger.info("🤖 Бот готов к работе. Запуск polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
            
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка при запуске бота: {e}", exc_info=True)

if __name__ == '__main__':
    # Импорт 'time' здесь, чтобы не засорять верхнюю часть файла
    import time
    main()