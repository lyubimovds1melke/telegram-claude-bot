import asyncio
import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import anthropic
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Получаем токены из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Проверяем наличие токенов
if not TELEGRAM_BOT_TOKEN or not ANTHROPIC_API_KEY:
    raise ValueError("Отсутствуют необходимые переменные окружения!")

# Инициализация клиента Anthropic
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

class ClaudeBot:
    def __init__(self):
        self.user_conversations = {}  # Хранение истории разговоров
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = """
🤖 Привет! Я бот с Claude Sonnet 4! ♥

Доступные команды:
/start - Показать это сообщение
/clear - Очистить историю разговора
/help - Помощь

Просто напишите мне сообщение, и я отвечу с помощью Claude Sonnet 4!
        """
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
🔧 Доступные команды:

/start - Начать работу с ботом
/clear - Очистить историю разговора
/help - Показать эту справку

💡 Советы:
• Задавайте любые вопросы на русском или английском
• Бот помнит контекст разговора
• Используйте /clear для сброса контекста
        """
        await update.message.reply_text(help_text)
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Очистка истории разговора"""
        user_id = update.effective_user.id
        if user_id in self.user_conversations:
            del self.user_conversations[user_id]
        await update.message.reply_text("✅ История разговора очищена!")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка обычных сообщений"""
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Отправляем индикатор печати
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Получаем или создаем историю разговора
            if user_id not in self.user_conversations:
                self.user_conversations[user_id] = []
            
            # Добавляем сообщение пользователя в историю
            self.user_conversations[user_id].append({
                "role": "user", 
                "content": user_message
            })
            
            # Ограничиваем историю (последние 10 сообщений)
            if len(self.user_conversations[user_id]) > 20:
                self.user_conversations[user_id] = self.user_conversations[user_id][-20:]
            
            # Отправляем запрос к Claude
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10000
                messages=self.user_conversations[user_id]
            )
            
            # Получаем ответ
            claude_response = response.content[0].text
            
            # Добавляем ответ в историю
            self.user_conversations[user_id].append({
                "role": "assistant", 
                "content": claude_response
            })
            
            # Отправляем ответ пользователю
            await update.message.reply_text(claude_response)
            
        except anthropic.RateLimitError:
            await update.message.reply_text("⚠️ Превышен лимит запросов. Попробуйте позже.")
        except anthropic.APIError as e:
            await update.message.reply_text(f"❌ Ошибка API: {str(e)}")
        except Exception as e:
            logging.error(f"Ошибка: {e}")
            await update.message.reply_text("❌ Произошла ошибка. Попробуйте еще раз.")

def main():
    """Главная функция запуска бота"""
    # Создаем экземпляр бота
    bot = ClaudeBot()
    
    # Создаем приложение Telegram
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", bot.start_command))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("clear", bot.clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # Запускаем бота
    print("🚀 Бот запускается...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()