import asyncio
import logging
import os
import time # Для time.monotonic() и time.sleep()
from typing import Dict, List, Any, AsyncGenerator

from datetime import datetime, timedelta

from telegram import Update, Message
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, SafetySetting, HarmCategory, Part, HarmBlockThreshold
# from google.api_core.exceptions import InvalidArgument # Необязателен, т.к. есть явная проверка на пустую историю

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
    SYSTEM_INSTRUCTION = os.getenv(
        "SYSTEM_INSTRUCTION",
        "Ты — дружелюбный и полезный ассистент Gemini в Telegram. Отвечай на русском языке."
    )
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "700000"))
    MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "100"))
    MAX_MESSAGE_LENGTH = 4096 # Лимит Telegram (пока не используется для разбивки ответа)
    RATE_LIMIT_MINUTES = int(os.getenv("RATE_LIMIT_MINUTES", "1"))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")

    @classmethod
    def validate(cls):
        if not cls.TELEGRAM_BOT_TOKEN or not cls.GEMINI_API_KEY:
            raise ValueError("❌ Отсутствуют обязательные переменные окружения: TELEGRAM_BOT_TOKEN, GEMINI_API_KEY")
        logger.info("✅ Конфигурация валидна")
        logger.info(f"Используемая модель Gemini: {cls.GEMINI_MODEL_NAME}")
        logger.info(f"Макс. токенов контекста истории: {cls.MAX_CONTEXT_TOKENS}")

# --- Управление данными ---
class RateLimiter:
    def __init__(self):
        self.user_requests: Dict[int, List[datetime]] = {}

    def is_allowed(self, user_id: int) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(minutes=Config.RATE_LIMIT_MINUTES)
        timestamps = self.user_requests.get(user_id, [])
        
        valid_timestamps = [t for t in timestamps if t > cutoff]
        
        if len(valid_timestamps) >= Config.RATE_LIMIT_REQUESTS:
            self.user_requests[user_id] = valid_timestamps # Обновляем, чтобы удалить старые, даже если лимит превышен
            return False
        
        valid_timestamps.append(now)
        self.user_requests[user_id] = valid_timestamps
        return True

    def cleanup_old_data(self):
        cutoff = datetime.now() - timedelta(hours=1)
        users_before_cleanup = len(self.user_requests)
        
        cleaned_requests: Dict[int, List[datetime]] = {}
        for user_id, timestamps in self.user_requests.items():
            valid_timestamps = [t for t in timestamps if t > cutoff]
            if valid_timestamps: # Только если остались актуальные записи
                cleaned_requests[user_id] = valid_timestamps
        
        self.user_requests = cleaned_requests
        users_after_cleanup = len(self.user_requests)
        
        if users_before_cleanup > 0 : # Логируем, только если были пользователи для очистки
            if users_before_cleanup != users_after_cleanup:
                logger.info(f"🧹 RateLimiter: количество отслеживаемых пользователей изменилось с {users_before_cleanup} на {users_after_cleanup} после очистки.")
            else:
                 logger.info(f"🧹 RateLimiter: проверена и очищена история запросов для {users_before_cleanup} пользователей (количество пользователей не изменилось).")


class ConversationManager:
    def __init__(self):
        self.conversations: Dict[int, List[Dict[str, Any]]] = {}
        self.last_activity: Dict[int, datetime] = {}

    def add_message(self, user_id: int, role: str, parts: List[Part]):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append({"role": role, "parts": parts})
        self.last_activity[user_id] = datetime.now()

        if len(self.conversations[user_id]) > Config.MAX_CONVERSATION_MESSAGES:
            self.conversations[user_id] = self.conversations[user_id][-Config.MAX_CONVERSATION_MESSAGES:]

    def get_conversation(self, user_id: int) -> List[Dict[str, Any]]:
        return self.conversations.get(user_id, [])

    def clear_conversation(self, user_id: int):
        if user_id in self.conversations:
            del self.conversations[user_id]
        if user_id in self.last_activity:
            del self.last_activity[user_id]
        logger.info(f"🧹 История разговора для пользователя {user_id} очищена.")

    def cleanup_inactive_conversations(self):
        cutoff = datetime.now() - timedelta(hours=24)
        to_remove = [uid for uid, t in self.last_activity.items() if t < cutoff]
        if to_remove: # Логируем только если есть что удалять
            for uid in to_remove:
                self.clear_conversation(uid) # clear_conversation уже логирует удаление
            logger.info(f"🧹 ConversationManager: завершена очистка {len(to_remove)} неактивных разговоров.")


# --- Основной класс бота ---
class GeminiBot:
    def __init__(self):
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
        except Exception as e:
            logger.critical(f"❌ Не удалось сконфигурировать Gemini API: {e}", exc_info=True)
            raise

        self.model = genai.GenerativeModel(
            model_name=Config.GEMINI_MODEL_NAME,
            system_instruction=Config.SYSTEM_INSTRUCTION,
            generation_config=GenerationConfig( # type: ignore[call-arg] # MyPy может ругаться на kwargs в TypedDict-like классах
                temperature=0.75,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        self.conversation_manager = ConversationManager()
        self.rate_limiter = RateLimiter()
        logger.info(f"🤖 Модель Gemini '{Config.GEMINI_MODEL_NAME}' инициализирована.")

    async def post_init(self, application: Application):
        asyncio.create_task(self._periodic_cleanup())
        logger.info("🛠️ Периодическая очистка данных запущена.")

    async def _periodic_cleanup(self):
        while True:
            await asyncio.sleep(3600) # Каждый час
            try:
                logger.info("⏳ Запуск периодической очистки...")
                self.conversation_manager.cleanup_inactive_conversations()
                self.rate_limiter.cleanup_old_data()
                logger.info("✅ Периодическая очистка завершена.")
            except Exception as e:
                logger.error(f"❌ Ошибка при периодической очистке: {e}", exc_info=True)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message: return
        user = update.effective_user
        user_name = user.first_name if user else "друг"
        await update.message.reply_text(
            f"👋 Привет, {user_name}! Я бот на базе **Google {Config.GEMINI_MODEL_NAME}**.\n\n"
            "Я помню большой контекст нашего разговора и могу анализировать изображения! "
            "Просто отправь мне картинку с текстом (или без).\n\n"
            "Используй /clear, чтобы начать разговор с чистого листа.",
            parse_mode=ParseMode.MARKDOWN
        )

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message: return
        if update.effective_user:
            self.conversation_manager.clear_conversation(update.effective_user.id)
            await update.message.reply_text("✅ История разговора очищена!")
        else:
            logger.warning("Получена команда /clear без effective_user.")
            await update.message.reply_text("Не удалось определить пользователя для очистки истории.")


    async def _prune_history_by_tokens(self, user_id: int) -> List[Dict[str, Any]]:
        history = self.conversation_manager.get_conversation(user_id)
        if not history:
            return []

        initial_message_count = len(history)
        current_total_tokens = 0 # Инициализируем на случай, если цикл не выполнится

        while True:
            try:
                if not history: # Если история стала пустой в процессе обрезки
                    current_total_tokens = 0
                else:
                    # model.count_tokens синхронный, оборачиваем в to_thread
                    # В google-generativeai==0.5.0 count_tokens(empty_list) вызывает ошибку,
                    # поэтому if not history: current_total_tokens = 0 выше это обрабатывает.
                    token_count_response = await asyncio.to_thread(self.model.count_tokens, history)
                    current_total_tokens = token_count_response.total_tokens

            except Exception as e:
                logger.error(f"Ошибка при подсчете токенов для user_id {user_id}: {e}", exc_info=True)
                break # Прерываем цикл, если не можем посчитать токены, возвращаем историю как есть

            if current_total_tokens <= Config.MAX_CONTEXT_TOKENS:
                break

            if len(history) > 1:
                history.pop(0) # Удаляем самое старое сообщение
            else: # Осталось одно сообщение, но оно все еще превышает лимит (или лимит слишком мал)
                  # Не удаляем его, пусть модель попробует обработать или вернет ошибку.
                logger.warning(
                    f"История для {user_id} содержит одно сообщение ({current_total_tokens} токенов), "
                    f"которое превышает MAX_CONTEXT_TOKENS ({Config.MAX_CONTEXT_TOKENS}) или "
                    f"не может быть далее урезано. Отправка как есть."
                )
                break
        
        final_message_count = len(history)
        if final_message_count < initial_message_count:
            logger.info(
                f"История для {user_id} урезана с {initial_message_count} до {final_message_count} сообщений. "
                f"Токены после урезки: {current_total_tokens}."
            )
        
        self.conversation_manager.conversations[user_id] = history
        return history

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.effective_user:
            logger.debug("Получено обновление без сообщения или пользователя.")
            return

        user_id = update.effective_user.id
        message = update.message
        
        if not self.rate_limiter.is_allowed(user_id):
            await message.reply_text("⏰ Превышен лимит запросов. Попробуйте чуть позже.", quote=True)
            return

        parts: List[Part] = []
        text_content = ""

        # Сначала извлекаем текст, потом фото. Порядок важен для Gemini.
        if message.text:
            text_content = message.text
        
        if message.photo:
            try:
                # Изображение сначала
                photo_file = await message.photo[-1].get_file()
                photo_bytes = await photo_file.download_as_bytearray()
                parts.append(Part.from_data(data=bytes(photo_bytes), mime_type="image/jpeg"))
                
                # Подпись к фото (если есть) добавляется как отдельная текстовая часть после изображения
                if message.caption:
                    text_content = message.caption # Используем caption как text_content, если он есть
                
                # Если был и обычный текст (маловероятно с фото, но для общности),
                # и подпись, подпись имеет приоритет для text_content
                # Если есть текст_контент (из message.text или message.caption) - добавляем его
                if text_content:
                    parts.append(Part.from_text(text_content))

            except Exception as e:
                logger.error(f"Ошибка при обработке фото для {user_id}: {e}", exc_info=True)
                await message.reply_text("😕 Не удалось обработать изображение. Попробуйте еще раз.", quote=True)
                return
        elif text_content: # Только текст (если message.photo было пустым)
            parts.append(Part.from_text(text_content))

        if not parts:
            logger.info(f"Получено пустое или неподдерживаемое сообщение от {user_id} (нет текста или фото).")
            # Можно отправить ответ пользователю или просто проигнорировать
            # await message.reply_text("Пожалуйста, отправьте текстовое сообщение или изображение.", quote=True)
            return

        placeholder_message = None # Инициализируем для блока finally/except
        try:
            await context.bot.send_chat_action(chat_id=user_id, action="typing")
            
            self.conversation_manager.add_message(user_id, "user", parts)
            pruned_history = await self._prune_history_by_tokens(user_id)
            
            # Важно: если pruned_history пуст, generate_content_async может вызвать ошибку.
            # Но наша логика add_message + _prune_history_by_tokens должна гарантировать,
            # что pruned_history содержит хотя бы текущее сообщение пользователя.
            if not pruned_history:
                logger.error(f"Критическая ошибка: pruned_history пуст для user {user_id} перед вызовом API.")
                await message.reply_text("Произошла внутренняя ошибка с историей сообщений. Пожалуйста, /clear и попробуйте снова.", quote=True)
                return

            placeholder_message = await message.reply_text("🧠 Думаю...", quote=True)
            
            response_stream = await self.model.generate_content_async(
                contents=pruned_history, # Явно указываем параметр contents
                stream=True
            )
            
            await self._stream_and_edit_message(response_stream, placeholder_message, user_id)

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке сообщения для {user_id}: {e}", exc_info=True)
            error_message_text = "❌ Произошла ошибка при обработке вашего запроса."
            # Попытка определить специфичные ошибки Gemini
            str_e = str(e).lower()
            if "safety setting" in str_e or "blocked" in str_e or "permission_denied" in str_e or "resource_exhausted" in str_e: # Общие маркеры проблем с API
                error_message_text = "К сожалению, я не могу ответить на этот запрос из-за ограничений или временных проблем с доступом. Попробуйте переформулировать или /clear."
            elif "quota" in str_e:
                error_message_text = " достигнут лимит запросов к API Gemini. Пожалуйста, попробуйте позже."
            
            try:
                if placeholder_message:
                    await placeholder_message.edit_text(error_message_text)
                else:
                    await message.reply_text(error_message_text, quote=True)
            except Exception as send_error:
                 logger.error(f"❌ Не удалось отправить/отредактировать сообщение об ошибке пользователю {user_id}: {send_error}")


    async def _stream_and_edit_message(self, stream: AsyncGenerator, tg_message: Message, user_id: int):
        full_response = ""
        buffer = ""
        last_edit_time = time.monotonic()
        edit_interval = 1.2  # секунды
        min_buffer_len_for_edit = 50 

        try:
            async for chunk in stream:
                # Безопасное получение текста из чанка (вдруг чанк без .text)
                text_part = getattr(chunk, 'text', None)
                if text_part:
                    buffer += text_part
                    full_response += text_part
                
                current_time = time.monotonic()
                if (current_time - last_edit_time > edit_interval and buffer) or len(buffer) >= min_buffer_len_for_edit :
                    if not buffer.strip() and not full_response.strip(): # Не редактируем, если пока только пробелы
                        continue
                    try:
                        await tg_message.edit_text(full_response + "█", parse_mode=ParseMode.MARKDOWN)
                        last_edit_time = current_time
                        buffer = "" # Сбрасываем буфер после успешного редактирования
                    except BadRequest as e:
                        if "Message is not modified" not in str(e).lower(): # Игнорируем только эту ошибку
                            logger.warning(f"Ошибка BadRequest при редактировании сообщения (user {user_id}): {e} | Текст: '{full_response + '█'}'")
                        # Если сообщение не изменилось, сбрасывать буфер не нужно, чтобы не потерять текст
                    except Exception as e:
                        logger.warning(f"Ошибка при редактировании сообщения (user {user_id}): {e}")
            
            # Финальное редактирование
            if full_response.strip(): # Отправляем, только если есть непустой текст
                try:
                    await tg_message.edit_text(full_response, parse_mode=ParseMode.MARKDOWN)
                except BadRequest as e:
                    if "Message is not modified" not in str(e).lower():
                        logger.warning(f"Ошибка BadRequest при финальном редактировании (user {user_id}): {e} | Текст: '{full_response}'")
                except Exception as e:
                     logger.warning(f"Ошибка при финальном редактировании (user {user_id}): {e}")
                self.conversation_manager.add_message(user_id, "model", [Part.from_text(full_response)])
            elif not full_response and getattr(stream, '_done_iterating', False) : # Стрим завершился, но ответа нет
                logger.info(f"Модель вернула пустой ответ (или только safety) для user {user_id}.")
                await tg_message.edit_text("😕 Ответ не был получен или был отфильтрован.", parse_mode=ParseMode.MARKDOWN)
            # Если full_response пустой, но стрим не завершился (ошибка в цикле), то ничего не делаем здесь,
            # ошибка должна была быть обработана выше.

        except Exception as e: # Ошибка в самом процессе стриминга (например, обрыв соединения с API)
            logger.error(f"❌ Ошибка в процессе получения или обработки стрима ответа для user {user_id}: {e}", exc_info=True)
            try:
                # Если был какой-то частичный ответ, можно его показать
                error_display_text = "❌ Произошла ошибка при получении полного ответа."
                if full_response.strip():
                    error_display_text += f"\nЧастичный ответ:\n{full_response[:1000]}" # Показать часть, если есть
                await tg_message.edit_text(error_display_text, parse_mode=ParseMode.MARKDOWN)
            except Exception as final_edit_error:
                logger.error(f"❌ Не удалось обновить сообщение об ошибке стриминга для user {user_id}: {final_edit_error}")


# --- Запуск бота ---
def main():
    try:
        logger.info(f"🚀 Запуск {Config.GEMINI_MODEL_NAME} Telegram Bot...")
        Config.validate()
        
        bot_instance = GeminiBot()
        
        application = (Application.builder()
                       .token(Config.TELEGRAM_BOT_TOKEN) # type: ignore[arg-type]
                       .concurrent_updates(10) # Можно настроить количество параллельных обработчиков
                       .post_init(bot_instance.post_init)
                       .build())
        
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.PHOTO) & ~filters.COMMAND,
            bot_instance.handle_message
        ))
        application.add_handler(CommandHandler("start", bot_instance.start_command))
        application.add_handler(CommandHandler("clear", bot_instance.clear_command))
        
        logger.info("🤖 Бот готов к работе. Запуск polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
            
    except ValueError as ve:
        logger.critical(f"❌ Ошибка конфигурации: {ve}", exc_info=True)
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка при запуске бота: {e}", exc_info=True)

if __name__ == '__main__':
    main()