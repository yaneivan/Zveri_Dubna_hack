import telebot
import os

# Вставьте сюда свой токен
TOKEN = '1886756451:AAGvgfHgeuMu07oiKlTZoPtgDHeqb-lhlrk'

bot = telebot.TeleBot(TOKEN)

# Определяем функцию для обработки команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 'Привет! Я простой бот. Отправь мне сообщение или фотографию, и я отвечу!')

# Определяем функцию для обработки текстовых сообщений
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)

# Определяем функцию для обработки фотографий
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Получаем информацию о фотографии
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Определяем путь для сохранения фотографии
    save_path = os.path.join(r'C:\Users\admin\Desktop', f'{message.photo[-1].file_id}.jpg')

    # Сохраняем фотографию на рабочем столе
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(message, 'Фотография сохранена на рабочем столе!')

# Запускаем бота
bot.polling()
