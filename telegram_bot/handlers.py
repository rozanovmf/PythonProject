import os
from pathlib import Path
from typing import Final
import tempfile
import cv2
import numpy as np

from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, FSInputFile, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from states import PhotoAnalysis
from src.flash_analyzer import FlashAnalyzer

router = Router()
analyzer: Final = FlashAnalyzer()


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "🌌 <b>Детектор световых вспышек</b>\n\n"
        "Отправь фото неба → я спрошу порог яркости → "
        "выделю объекты рамками и дам статистику.\n\n"
        "<i>Просто пришли фотографию</i>",
        parse_mode="HTML"
    )


@router.message(F.photo)
async def got_photo(message: Message, state: FSMContext):
    photo = message.photo[-1]

    file = await message.bot.get_file(photo.file_id)
    bytes_data = await message.bot.download_file(file.file_path)

    # Сохраняем во временный файл
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"temp_photo_{message.from_user.id}.jpg")

    with open(temp_path, "wb") as f:
        f.write(bytes_data.read())

    # Проверяем, что изображение загрузилось корректно
    test_img = cv2.imread(temp_path)
    if test_img is None:
        await message.answer("❌ Не удалось прочитать изображение. Попробуйте другое фото.")
        return

    h, w = test_img.shape[:2]
    await message.answer(
        f"📸 Фото получено! Размер: {w}x{h}\n\n"
        f"Теперь выберите чувствительность детекции:",
        reply_markup=get_sensitivity_keyboard()
    )

    await state.update_data(temp_photo_path=temp_path)
    await state.set_state(PhotoAnalysis.waiting_for_threshold)


def get_sensitivity_keyboard():
    """Клавиатура с предустановленными значениями чувствительности"""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🔍 Высокая (0.005)"),
                KeyboardButton(text="⭐ Средняя (0.01)")
            ],
            [
                KeyboardButton(text="✨ Низкая (0.05)"),
                KeyboardButton(text="💫 Очень низкая (0.1)")
            ],
            [
                KeyboardButton(text="✏️ Ввести своё значение")
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )


@router.message(PhotoAnalysis.waiting_for_threshold)
async def process_threshold(message: Message, state: FSMContext):
    text = message.text.strip()

    # Обработка предустановленных значений
    threshold_map = {
        "🔍 Высокая (0.005)": 0.005,
        "⭐ Средняя (0.01)": 0.01,
        "✨ Низкая (0.05)": 0.05,
        "💫 Очень низкая (0.1)": 0.1,
    }

    if text in threshold_map:
        threshold = threshold_map[text]
    elif text == "✏️ Ввести своё значение":
        await message.answer(
            "Введите число от 0.001 до 0.5\n"
            "💡 Чем меньше число, тем больше объектов будет найдено\n\n"
            "Примеры:\n"
            "• 0.005 — очень высокая чувствительность\n"
            "• 0.01 — высокая чувствительность\n"
            "• 0.05 — средняя чувствительность\n"
            "• 0.1 — низкая чувствительность",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    else:
        try:
            threshold = float(text)
            if not 0.001 <= threshold <= 0.5:
                raise ValueError
        except Exception:
            await message.answer(
                "❌ Пожалуйста, введите число от 0.001 до 0.5\n"
                "Например: 0.005, 0.01, 0.05",
                reply_markup=get_sensitivity_keyboard()
            )
            return

    data = await state.get_data()
    photo_path = data.get("temp_photo_path")

    if not photo_path or not os.path.exists(photo_path):
        await message.answer(
            "❌ Ошибка: фото не найдено. Попробуй отправить заново.",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.clear()
        return

    await message.answer(
        f"🔍 Чувствительность: <b>{threshold:.3f}</b>\n"
        "Анализирую изображение... (это может занять 5–15 секунд)\n\n"
        "⏳ Пожалуйста, подождите...",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardRemove()
    )

    try:
        # Анализируем изображение
        saved_files, stats = analyzer.analyze_real_sequence(
            source_path=photo_path,
            n_frames=1,
            threshold=threshold
        )

        if not saved_files:
            # Если ничего не найдено, предлагаем другие значения
            await message.answer(
                "🔍 <b>Ничего не найдено</b>\n\n"
                "Попробуйте:\n"
                "• Выбрать более высокую чувствительность (меньшее число)\n"
                "• Отправить фото с более контрастным небом\n"
                "• Использовать фото с более яркими объектами\n\n"
                "Отправьте новое фото или нажмите /start",
                parse_mode="HTML"
            )
        else:
            # Отправляем найденные аннотированные изображения
            sent_count = 0
            for file_path in saved_files[:5]:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    file_name = Path(file_path).name
                    await message.answer_photo(
                        photo=FSInputFile(file_path),
                        caption=f"📷 Результат анализа"
                    )
                    sent_count += 1

            if sent_count == 0:
                await message.answer("⚠️ Не удалось отправить результаты анализа.")

            # Отправляем статистику
            if stats and stats.get("total_objects", 0) > 0:
                stats_text = "<b>📊 Результаты анализа:</b>\n"
                stats_text += "━━━━━━━━━━━━━━━━━━━━━━\n"
                stats_text += f"📁 Обработано: {stats['frames_processed']} кадр(ов)\n"
                stats_text += f"✨ Найдено объектов: {stats['total_objects']}\n\n"

                if stats.get("by_type"):
                    stats_text += "<b>🔭 Обнаруженные объекты:</b>\n"
                    for typ, cnt in stats["by_type"].items():
                        emoji = {
                            "звезда (точечная)": "⭐",
                            "звезда с дифракцией / кластер": "🌟",
                            "галактика / крупный кластер": "🌌",
                            "вспышка / спутник / мусор / артефакт": "💥",
                            "вспышка (вытянутый)": "✨",
                            "мусор / артефакт": "🗑️",
                            "неизвестно": "❓"
                        }.get(typ, "•")
                        stats_text += f"{emoji} {typ}: {cnt} шт.\n"

                await message.answer(stats_text, parse_mode="HTML")

    except Exception as e:
        error_msg = str(e)
        await message.answer(
            f"❌+ <b>Произошла ошибка:</b>\n"
            f"<code>{error_msg[:200]}</code>\n\n"
            f"Попробуйте:\n"
            f"• Отправить другое фото\n"
            f"• Выбрать другую чувствительность\n"
            f"• Связаться с разработчиком",
            parse_mode="HTML"
        )
        print(f"Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Удаляем временный файл
        if os.path.exists(photo_path):
            try:
                os.unlink(photo_path)
            except Exception as e:
                print(f"Не удалось удалить {photo_path}: {e}")

        # Сбрасываем состояние
        await state.clear()