import os
import base64
import requests
import json
import pandas as pd
from geopy.distance import geodesic
from mistralai import Mistral
import telebot
import logging
import traceback
from queue import Queue, Empty
from threading import Thread, Event

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API ключ для Vision LLM
api_key = "BsoGgWX10TneaJMu89JevSiGCjU3ODnG"

# Функция для кодирования изображения в base64
def encode_image(image_path):
    logger.info(f"Начало кодирования изображения: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            logger.info("Изображение успешно закодировано")
            return encoded
    except FileNotFoundError:
        logger.error(f"Файл не найден: {image_path}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Ошибка при кодировании изображения: {e}")
        logger.error(traceback.format_exc())
        return None

# Функция для создания JSON промпта для анализа текста
def create_json_prompt_for_text_analysis(text):
    logger.info("Создание JSON промпта для анализа текста")
    try:
        prompt = {
            "text": text,
            "entities": [
                {"entity": "address", "types": ["address"]},
                {"entity": "animal type", "types": ["animal type"]},
                {"entity": "status", "types": ["status"]},
                {"entity": "description", "types": ["description"]}
            ]
        }
        return json.dumps(prompt)
    except Exception as e:
        logger.error(f"Ошибка при создании JSON промпта: {e}")
        logger.error(traceback.format_exc())
        return None

# Функция для взаимодействия с Vision LLM
def get_vision_llm_response(text):
    logger.info("Отправка запроса к Vision LLM")
    try:
        model = "pixtral-12b-2409"
        client = Mistral(api_key=api_key)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
**Objective:**
Analyze the given text and extract the address, animal type, status, and description.

**Format Requirements:**
- Do not explain your answer. Just output the JSON.

**Entity Annotation Details:**
- The entity "animal type" must be either "Кошка" or "Собака".
- The entity "status" must be either "Найдена" or "Пропала".
- Entity "address" можно редактировать, так, чтобы в итогде получился запрос адреса для гугл карт. (например можно убрать склонения)

**Example:**

text = "Пропала собака, улица Ленина, дом 5, Санкт-Петербург. Очень дружелюбная собака, черного цвета, пропала во дворе, возможно, выбежала на улицу."

{{
    "Адрес": "Санкт-Петербург, улица Ленина, дом 5",
    "Тип животного": "Собака",
    "Статус": "Пропала",
    "Описание": "Очень дружелюбная собака, черного цвета, пропала во дворе, возможно, выбежала на улицу."
}}

Text:
"{text}"
"""
                    }
                ]
            }
        ]

        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        logger.info("Получен ответ от Vision LLM")
        return chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка при взаимодействии с Vision LLM: {e}")
        logger.error(traceback.format_exc())
        return None

# Функция для геокодирования адреса с использованием Nominatim API
def geocode_address(address):
    logger.info(f"Геокодирование адреса: {address}")
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'PetFinderApp/1.0 (anton.epub@gmail.com)'
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if (data):
            location = data[0]
            latitude = float(location['lat'])
            longitude = float(location['lon'])
            logger.info(f"Получены координаты: {latitude}, {longitude}")
            return latitude, longitude
        else:
            logger.warning("Адрес не найден")
            return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ошибка HTTP: {e}")
        logger.error(f"Текст ответа: {response.text}")
        logger.error(traceback.format_exc())
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса: {e}")
        logger.error(traceback.format_exc())
        return None
    except ValueError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        logger.error(f"Текст ответа: {response.text}")
        logger.error(traceback.format_exc())
        return None

# Функция для сравнения животного с описанием
def compare_animal(record, target_description, user_images=None):
    logger.info(f"Сравнение животного с описанием: {target_description}")
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'Сравните двух животных. Нужно определить, первое животное это то же самое что и второе, либо это разные животные. В конце твоего сообщения должен быть итоговый ответ.\n\n Наченем. Описание первого животного: {target_description}\nФотографии первого животного:'
                    }
                ]
            }
        ]

        # Добавляем фотографии из базы данных
        for img in record['imgs']:
            if img is not None:
                messages[0]['content'].append({
                    "type": "image_url",
                    "image_url": img
                })
            else:
                logger.warning("Обнаружена ссылка на изображение: None")

        messages[0]['content'].append({
            "type": "text",
            "text": f"\n\nОписание второго животного: {record['description']}\nФотографии второго животного:"
        })

        # Добавляем фотографии пользователя, если они есть
        if user_images:
            for img_path in user_images:
                base64_image = encode_image(img_path)
                if base64_image:
                    messages[0]['content'].append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    })
        else:
            messages[0]['content'].append({
                "type": "text",
                "text": f"Нет фотографий. "
            })

        messages[0]['content'].append({
            "type": "text",
            "text": f'После рассуждений, напиши итоговые ответ в формате: \n\nОтвет: ```json {{"result": "похожее объявление" или "другое животное"}}```'
        })

        logger.info("Отправка запроса к LLM")
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        logger.info("Получен ответ от LLM")
        return chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка при сравнении животных: {e}")
        logger.error(f"Ошибка вызвана объявлением: {record['link']}, с описанием: {record['description']}")
        logger.error(f"Ссылки на изображения: {record['imgs']}")
        logger.error(traceback.format_exc())
        return None

# Инициализация бота
bot = telebot.TeleBot('1886756451:AAGvgfHgeuMu07oiKlTZoPtgDHeqb-lhlrk')

# Retrieve the API key from environment variables
api_key = "BsoGgWX10TneaJMu89JevSiGCjU3ODnG"

# Specify model
model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Создаем словарь для хранения временных данных пользователей
user_data = {}

# Создаем очередь для обработки запросов пользователей
user_queue = Queue()

# Функция для обработки команды /help
@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """
    🐾 Помощь по использованию бота:
    
    1. Отправьте описание потерянного или найденного животного
    2. Укажите в сообщении:
       - Тип животного (кошка/собака)
       - Статус (найдена/пропала)
       - Адрес
       - Описание животного
    3. Можете прикрепить фотографии животного (до 10 штук)
    
    Пример сообщения:
    "Пропала собака в районе ул. Ленина, 5. Черная, с белым пятном на груди, отзывается на кличку Рекс"
    
    /start - Начать поиск
    /help - Показать эту справку
    """
    bot.reply_to(message, help_text)
# Функция для обработки команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info(f"Получена команда /start от пользователя {message.from_user.id}")
    bot.reply_to(message, 'Привет! 🐾 Отправьте мне описание потерянного или найденного животного и фотографии (если есть). Для помощи используйте /help')
# Обработчик фотографий с подписью
@bot.message_handler(content_types=['photo'])
def handle_photos(message):
    user_id = message.from_user.id
    logger.info(f"Получено фото с подписью от пользователя {user_id}")
    
    # Создаем директорию для входящих изображений, если её нет
    if not os.path.exists('tg_input_imgs'):
        os.makedirs('tg_input_imgs')
    
    # Получаем информацию о фото
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    # Сохраняем фото
    image_path = f'tg_input_imgs/{user_id}_{file_info.file_path.split("/")[-1]}'
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    
    # Если есть подпись к фото, обрабатываем её как описание
    if message.caption:
        user_queue.put((message, message.caption, [image_path]))
    else:
        bot.reply_to(message, "Пожалуйста, добавьте описание к фотографии")

# Функция для обработки текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    logger.info(f"Получено сообщение от пользователя {user_id}: {message.text}")
    user_queue.put((message, message.text, None))

def handle_search_request(message, text, user_images=None):
    logger.info(f"Обработка поискового запроса от пользователя {message.from_user.id}")
    try:
        llm_response = get_vision_llm_response(text)
        logger.info(f"Ответ LLM: {llm_response}")

        try:
            llm_response = llm_response.replace("```json", "").replace("```", "")
            llm_data = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка декодирования JSON: {e}")
            logger.error(f"llm_response: {llm_response}")
            logger.error(traceback.format_exc())
            llm_data = {}

        address = llm_data.get('Адрес')
        animal_type = llm_data.get('Тип животного') if llm_data.get('Тип животного') in ['Кошка', 'Собака'] else None
        status = llm_data.get('Статус') if llm_data.get('Статус') in ['Найдена', 'Пропала'] else None
        description = llm_data.get('Описание')

        logger.info(f"Извлеченные данные: Адрес={address}, Тип={animal_type}, Статус={status}")

        try:
            links_df = pd.read_csv('links.csv')
            links_df['imgs'] = links_df['imgs'].apply(eval).tolist()
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из CSV: {e}")
            logger.error(traceback.format_exc())
            bot.reply_to(message, "Произошла ошибка при обработке данных 😿")
            return

        if animal_type and status:
            filtered_data = pd.DataFrame()
            if animal_type == 'Кошка':
                if status == 'Пропала':
                    filtered_data = links_df[(links_df['status'] == 'Найдена') & (links_df['animal_type'] == 'Кошка')]
                elif status == 'Найдена':
                    filtered_data = links_df[(links_df['status'] == 'Пропала') & (links_df['animal_type'] == 'Кошка')]
            elif animal_type == 'Собака':
                if status == 'Пропала':
                    filtered_data = links_df[(links_df['status'] == 'Найдена') & (links_df['animal_type'] == 'Собака')]
                elif status == 'Найдена':
                    filtered_data = links_df[(links_df['status'] == 'Пропала') & (links_df['animal_type'] == 'Собака')]
            if address:
                coordinates = geocode_address(address)
                if coordinates:
                    latitude, longitude = coordinates
                    logger.info(f"Координаты: {latitude}, {longitude}")

                    # Проверяем координаты на валидность перед вычислением расстояний
                    if pd.isna(latitude) or pd.isna(longitude):
                        logger.warning("Получены невалидные координаты")
                        bot.reply_to(message, "Не удалось определить точные координаты для данного адреса 😿")
                        return

                    # Отфильтровываем записи с невалидными координатами
                    filtered_data = filtered_data.dropna(subset=['latitude', 'longitude'])

                    # Вычисляем расстояния только для записей с валидными координатами
                    filtered_data['distance'] = filtered_data.apply(
                        lambda row: geodesic((latitude, longitude), (row['latitude'], row['longitude'])).kilometers 
                        if not pd.isna(row['latitude']) and not pd.isna(row['longitude'])
                        else float('inf'),
                        axis=1
                    )
                    sorted_data = filtered_data.sort_values(by='distance')

                    for index, record in sorted_data.head(15).iterrows():
                        logger.info(f"Обработка записи {index + 1}: {record['title']}")
                        result = compare_animal(record, description, user_images)
                        
                        if result:
                            json_string = result.split("Ответ:")[-1].strip().replace('```json', '').replace('```', '')
                            attempts = 0
                            json_result = None
                            while attempts < 3:
                                try:
                                    json_result = json.loads(json_string)
                                    break
                                except json.JSONDecodeError:
                                    attempts += 1
                                    logger.warning(f"Попытка {attempts}: ошибка декодирования JSON")
                            
                            if json_result is None:
                                if 'похожее объявление' in result and not("другое животное" in result):
                                    json_result = {'result': 'похожее объявление'}
                                else:
                                    continue

                            if json_result['result'] == "похожее объявление":
                                logger.info(f"Найдено похожее объявление: {record['link']}")
                                # Создаем список медиа с описанием в первом фото
                                media_group = []
                                first_photo = True
                                for img_url in record['imgs']:
                                    try:
                                        img_response = requests.get(img_url)
                                        img_dir = 'tg_imgs'
                                        if not os.path.exists(img_dir):
                                            os.makedirs(img_dir)
                                        img_path = os.path.join(img_dir, f"{record['link'].split('/')[-1]}_{len(media_group)}.jpg")
                                        with open(img_path, 'wb') as img_file:
                                            img_file.write(img_response.content)
                                                                                
                                        if first_photo:
                                            # К первому фото добавляем описание
                                            caption = (
                                                f"Результат: {json_result['result']} 🐾\n"
                                                f"Описание: {record['description']}\n"
                                                f"Адрес: {record['address']}\n"
                                                f"Ссылка: {record['link']}\n"
                                                f"GPS метка: {record['latitude']}, {record['longitude']}\n"
                                            )
                                            media_group.append(telebot.types.InputMediaPhoto(open(img_path, 'rb'), caption=caption))
                                            first_photo = False
                                        else:
                                            media_group.append(telebot.types.InputMediaPhoto(open(img_path, 'rb')))
                                    except Exception as e:
                                        logger.error(f"Ошибка при обработке изображения: {e}")
                                        logger.error(traceback.format_exc())
                                        continue
                                        
                                if media_group:
                                    bot.send_media_group(message.chat.id, media_group)
                                    logger.info(f"Отправлена группа изображений с описанием")
                            else:
                                logger.info(f"Объявление не похоже: {record['link']}\n\n")
                                if index == len(sorted_data.head(7)) - 1:
                                    bot.reply_to(message, "К сожалению, не нашлось похожих объявлений 😿🐾💔\nНо мы продолжим искать! 🔍✨\nПопробуйте проверить позже или измените описание 🌟")

                else:
                    logger.warning("Не удалось получить координаты")
                    bot.reply_to(message, "Не удалось получить координаты для данного адреса 😿")
            else:
                logger.warning("Адрес не указан")
                bot.reply_to(message, "Адрес не указан для геокодирования 😿")
        else:
            logger.warning("Неверный тип животного или статус")
            bot.reply_to(message, "Пожалуйста, укажите корректный тип животного (кошка/собака) и статус (найдена/пропала) 🐾")

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке сообщения: {e}")
        logger.error(f"message: {message}")
        logger.error(f"text: {text}")
        logger.error(f"user_images: {user_images}")
        logger.error(traceback.format_exc())
        bot.reply_to(message, "Произошла ошибка при обработке вашего сообщения. Пожалуйста, попробуйте позже 😿")

def process_queue(stop_event):
    while not stop_event.is_set():
        try:
            message, text, user_images = user_queue.get(timeout=1)
            handle_search_request(message, text, user_images)
            user_queue.task_done()
        except Empty:
            continue

# Запуск потока для обработки очереди
stop_event = Event()
queue_thread = Thread(target=process_queue, args=(stop_event,), daemon=True)
queue_thread.start()

# Запуск бота
if __name__ == '__main__':
    logger.info("Запуск бота")
    try:
        bot.polling()
    except KeyboardInterrupt:
        logger.info("Остановка бота")
        stop_event.set()
        queue_thread.join()
    except Exception as e:
        logger.critical(f"Критическая ошибка при работе бота: {e}")
        logger.critical(traceback.format_exc())
