import asyncio
import logging
import os
import time # Перенесен сюда
from typing import Dict, List, Any, AsyncGenerator
from datetime import datetime, timedelta

from telegram import Update, Message
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import BadRequest
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, SafetySetting, HarmCategory, Part, HarmBlockThreshold

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
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "700000")) # 700k токенов для истории
    MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "100")) # Лимит на число сообщений в истории
    MAX_MESSAGE_LENGTH = 4096 # Лимит Telegram (пока не используется для разбивки ответа)
    RATE_LIMIT_MINUTES = int(os.getenv("RATE_LIMIT_MINUTES", "1"))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20")) # Gemini 1.5 Pro позволяет больше запросов
    # Модель Gemini для использования
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
            self.user_requests[user_id] = valid_timestamps
            return False
        valid_timestamps.append(now)
        self.user_requests[user_id] = valid_timestamps
        return True

    def cleanup_old_data(self):
        cutoff = datetime.now() - timedelta(hours=1) # Очищаем данные старше 1 часа
        users_before_cleanup = len(self.user_requests)
        self.user_requests = {
            uid: ts
            for uid, ts in self.user_requests.items()
            if any(t > cutoff for t in ts)
        }
        cleaned_count = users_before_cleanup - len(self.user_requests)
        if cleaned_count > 0:
            logger.info(f"🧹 RateLimiter: очищены данные для {cleaned_count} пользователей.")

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
        if user_id in self.conversations:
            del self.conversations[user_id]
        if user_id in self.last_activity:
            del self.last_activity[user_id]
        logger.info(f"🧹 История разговора для пользователя {user_id} очищена.")


    def cleanup_inactive_conversations(self):
        cutoff = datetime.now() - timedelta(hours=24) # Очищаем разговоры старше 24 часов
        to_remove = [uid for uid, t in self.last_activity.items() if t < cutoff]
        for uid in to_remove:
            self.clear_conversation(uid) # Использует существующий метод, который также логгирует
        if to_remove:
            logger.info(f"🧹 ConversationManager: очищено {len(to_remove)} неактивных разговоров.")

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
            generation_config=GenerationConfig( # type: ignore
                temperature=0.75,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192, # Максимум для ответа
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

    # --- Команды ---
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        user_name = user.first_name if user else "друг"
        await update.message.reply_text(
            f"👋 Привет, {user_name}! Я бот на базе **Google Gemini {Config.GEMINI_MODEL_NAME}**.\n\n"
            "Я помню большой контекст нашего разговора и могу анализировать изображения! "
            "Просто отправь мне картинку с текстом (или без).\n\n"
            "Используй /clear, чтобы начать разговор с чистого листа.",
            parse_mode=ParseMode.MARKDOWN
        )

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user:
            self.conversation_manager.clear_conversation(update.effective_user.id)
            await update.message.reply_text("✅ История разговора очищена!")
        else:
            logger.warning("Получена команда /clear без effective_user.")


    async def _prune_history_by_tokens(self, user_id: int) -> List[Dict[str, Any]]:
        history = self.conversation_manager.get_conversation(user_id)
        if not history:
            return []

        initial_message_count = len(history)
        
        # Используем асинхронный подсчет токенов
        # Примечание: count_tokens не учитывает system_instruction, заданный в модели.
        # MAX_CONTEXT_TOKENS должен быть установлен с этим учётом.
        while True:
            try:
                # В `google-generativeai` версии 0.5.0+ `count_tokens` для `GenerativeModel`
                # принимает `contents` (историю) и возвращает объект с `total_tokens`.
                # Если `history` пуста, то `count_tokens` может вызвать ошибку или вернуть 0.
                if not history:
                    token_count_response = await asyncio.to_thread(self.model.count_tokens, []) # или просто 0
                    current_total_tokens = getattr(token_count_response, 'total_tokens', 0)

                else:
                    # `model.count_tokens` может быть синхронным, обернем в to_thread для безопасности
                    token_count_response = await asyncio.to_thread(self.model.count_tokens, history)
                    current_total_tokens = token_count_response.total_tokens

            except Exception as e:
                logger.error(f"Ошибка при подсчете токенов для user_id {user_id}: {e}", exc_info=True)
                # Если не можем посчитать токены, лучше вернуть историю как есть, чтобы не потерять данные
                # или вернуть пустую, если это критично. Здесь возвращаем как есть.
                break 

            if current_total_tokens <= Config.MAX_CONTEXT_TOKENS:
                break

            if len(history) > 1: # Удаляем старые сообщения, кроме самого последнего (если оно одно)
                history.pop(0)
            else: # Если даже одно сообщение слишком большое (но вписывается) или что-то пошло не так
                break
        
        final_message_count = len(history)
        if final_message_count < initial_message_count:
            logger.info(
                f"История для {user_id} урезана с {initial_message_count} до {final_message_count} сообщений. "
                f"Токены после урезки: {current_total_tokens if 'current_total_tokens' in locals() else 'N/A'}."
            )
        
        # Обновляем историю в менеджере разговоров только если она действительно изменилась
        # или если это необходимо для консистентности (например, если get_conversation возвращает копию)
        # В данном случае, get_conversation возвращает ссылку, так что изменение history уже отражено.
        # Но для явности можно сделать так:
        self.conversation_manager.conversations[user_id] = history
        return history

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.effective_user:
            return

        user_id = update.effective_user.id
        message = update.message
        
        if not self.rate_limiter.is_allowed(user_id):
            await message.reply_text("⏰ Превышен лимит запросов. Попробуйте чуть позже.")
            return

        parts: List[Part] = []
        text_content = message.text or message.caption or ""

        if message.photo:
            try:
                photo_file = await message.photo[-1].get_file()
                photo_bytes = await photo_file.download_as_bytearray()
                # Сначала изображение, потом текст (если есть) - для мультимодальных моделей это может иметь значение
                parts.append(Part.from_data(data=bytes(photo_bytes), mime_type="image/jpeg"))
                if text_content: # Если есть подпись к фото, она уже в text_content
                     parts.append(Part.from_text(text_content))
            except Exception as e:
                logger.error(f"Ошибка при обработке фото для {user_id}: {e}", exc_info=True)
                await message.reply_text("😕 Не удалось обработать изображение. Попробуйте еще раз.")
                return
        elif text_content: # Только текст
            parts.append(Part.from_text(text_content))

        if not parts: # Если нет ни текста, ни фото (например, стикер, аудио и т.д.)
            # Можно добавить ответ, что такие типы сообщений не поддерживаются, или просто проигнорировать
            logger.info(f"Получено пустое или неподдерживаемое сообщение от {user_id}.")
            return

        try:
            await context.bot.send_chat_action(chat_id=user_id, action="typing")
            
            self.conversation_manager.add_message(user_id, "user", parts)
            pruned_history = await self._prune_history_by_tokens(user_id)
            
            if not pruned_history and Config.SYSTEM_INSTRUCTION:
                # Если история пуста, но есть системная инструкция,
                # Gemini API может требовать хотя бы одно сообщение пользователя.
                # Однако, `add_message` уже добавил текущее сообщение пользователя.
                # Этот блок может быть не нужен, если `pruned_history` всегда будет содержать хотя бы текущее сообщение.
                # Если же `_prune_history_by_tokens` может вернуть пустой список, то:
                # logger.warning(f"История для {user_id} пуста после обрезки. Отправка только системной инструкции не поддерживается.")
                # await message.reply_text("Произошла ошибка с историей сообщений. Попробуйте /clear.")
                # return
                pass # Пропускаем, так как add_message уже добавил текущее сообщение.


            placeholder_message = await message.reply_text("🧠 Думаю...", quote=True)
            
            # Генерация контента со стримингом
            response_stream = await self.model.generate_content_async(
                pruned_history, # `pruned_history` уже содержит parts последнего сообщения
                stream=True
            )
            
            await self._stream_and_edit_message(response_stream, placeholder_message, user_id)

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке сообщения для {user_id}: {e}", exc_info=True)
            error_message = "❌ Произошла ошибка при обработке вашего запроса."
            if "blocked by safety setting" in str(e).lower():
                error_message = " maaf, saya tidak bisa menanggapi permintaan ini karena batasan keamanan." #Пример ответа на индонезийском, если сработают safety settings (лучше локализовать)
            elif "The HAP check failed" in str(e): # Специфичная ошибка Gemini, связанная с безопасностью
                 error_message = " maaf, saya tidak bisa menanggapi permintaan ini karena batasan keamanan."
            elif "quota" in str(e).lower():
                error_message = " достигнут лимит запросов к API Gemini. Пожалуйста, попробуйте позже."

            try:
                if 'placeholder_message' in locals() and placeholder_message:
                    await placeholder_message.edit_text(error_message)
                else:
                    await message.reply_text(error_message)
            except Exception as send_error:
                 logger.error(f"❌ Не удалось отправить сообщение об ошибке пользователю {user_id}: {send_error}")


    async def _stream_and_edit_message(self, stream: AsyncGenerator, tg_message: Message, user_id: int):
        full_response = ""
        buffer = ""
        last_edit_time = time.monotonic()
        edit_interval = 1.2  # секунды
        min_buffer_len_for_edit = 50 # Минимальная длина буфера для обновления, чтобы не слишком часто дергать API

        try:
            async for chunk in stream:
                if text_part := getattr(chunk, 'text', None): # Безопасное получение текста из чанка
                    buffer += text_part
                    full_response += text_part
                
                current_time = time.monotonic()
                if (current_time - last_edit_time > edit_interval and buffer) or len(buffer) > min_buffer_len_for_edit :
                    try:
                        # Используем MarkdownV2, если планируется сложная разметка, или оставляем MARKDOWN
                        # Для MarkdownV2 нужно экранировать спецсимволы: .!-_*[]()~`>#+=|{}
                        # Пока оставим ParseMode.MARKDOWN, он проще.
                        await tg_message.edit_text(full_response + "█", parse_mode=ParseMode.MARKDOWN)
                        last_edit_time = current_time
                        buffer = ""
                    except BadRequest as e:
                        if "Message is not modified" not in str(e):
                            logger.warning(f"Ошибка BadRequest при редактировании сообщения (user {user_id}): {e}")
                        # Если сообщение не изменилось, просто продолжаем
                    except Exception as e:
                        logger.warning(f"Ошибка при редактировании сообщения (user {user_id}): {e}")
                        # Продолжаем, чтобы не прерывать стриминг из-за временной ошибки Telegram

            # Финальное редактирование, чтобы убрать курсор и отправить полный текст
            if full_response:
                try:
                    await tg_message.edit_text(full_response, parse_mode=ParseMode.MARKDOWN)
                except BadRequest as e:
                    if "Message is not modified" not in str(e):
                        logger.warning(f"Ошибка BadRequest при финальном редактировании (user {user_id}): {e}")
                except Exception as e:
                     logger.warning(f"Ошибка при финальном редактировании (user {user_id}): {e}")

            else: # Если модель не вернула текст (например, из-за safety settings или пустой ответ)
                logger.info(f"Модель вернула пустой ответ для user {user_id}.")
                await tg_message.edit_text("😕 Ответ не был получен или был пустым.", parse_mode=ParseMode.MARKDOWN)
            
            # Добавляем ответ модели в историю, только если он не пустой
            if full_response:
                 self.conversation_manager.add_message(user_id, "model", [Part.from_text(full_response)])

        except Exception as e:
            logger.error(f"❌ Ошибка в процессе стриминга ответа для user {user_id}: {e}", exc_info=True)
            try:
                await tg_message.edit_text("❌ Произошла ошибка при получении ответа.", parse_mode=ParseMode.MARKDOWN)
            except Exception as final_edit_error:
                logger.error(f"❌ Не удалось обновить сообщение об ошибке стриминга для user {user_id}: {final_edit_error}")


# --- Запуск бота ---
def main():
    try:
        logger.info(f"🚀 Запуск {Config.GEMINI_MODEL_NAME} Telegram Bot...")
        Config.validate() # Валидация конфигурации перед инициализацией бота
        
        bot_instance = GeminiBot() # Инициализация бота после валидации конфига
        
        application = (Application.builder()
                       .token(Config.TELEGRAM_BOT_TOKEN) # type: ignore
                       .concurrent_updates(True) # Обработка нескольких апдейтов параллельно
                       .post_init(bot_instance.post_init)
                       .build())
        
        # Обработчик для текстовых сообщений и фото с подписями
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.PHOTO) & ~filters.COMMAND,
            bot_instance.handle_message
        ))
        application.add_handler(CommandHandler("start", bot_instance.start_command))
        application.add_handler(CommandHandler("clear", bot_instance.clear_command))
        
        # TODO: Добавить кастомный обработчик ошибок Telegram: application.add_error_handler(error_handler_callback)
        
        logger.info("🤖 Бот готов к работе. Запуск polling...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
            
    except ValueError as ve: # Ошибки валидации конфигурации
        logger.critical(f"❌ Ошибка конфигурации: {ve}", exc_info=True)
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка при запуске бота: {e}", exc_info=True)

if __name__ == '__main__':
    main()