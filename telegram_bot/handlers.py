import os
from pathlib import Path
from typing import Final
import tempfile

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

    await state.update_data(temp_photo_path=temp_path)

    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="0.01")]],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Введите число от 0.01 до 1.0"
    )

    await message.answer(
        "📸 Фото получено!\n\n"
        "Введите порог яркости для выделения объектов от 0.01 до 1.0\n"
        "💡 <b>Рекомендация:</b> 0.01-0.05 для тёмного неба, 0.05-0.15 для светлого\n\n"
        "<i>Меньшее значение = больше объектов будет найдено</i>",
        parse_mode="HTML",
        reply_markup=keyboard
    )
    await state.set_state(PhotoAnalysis.waiting_for_threshold)


@router.message(PhotoAnalysis.waiting_for_threshold)
async def process_threshold(message: Message, state: FSMContext):
    try:
        threshold = float(message.text.strip())
        if not 0.01 <= threshold <= 1.0:
            raise ValueError
    except Exception:
        await message.answer(
            "❌ Пожалуйста, введи число от 0.01 до 1.0\n"
            "Например: 0.01, 0.05, 0.1"
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
        f"🔍 Порог: <b>{threshold:.3f}</b>\n"
        "Анализирую изображение... (это может занять 5–15 секунд)",
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

        output_dir = Path("annotated_frames")

        if not saved_files:
            await message.answer(
                "🔍 Ничего интересного не найдено с таким порогом.\n"
                "💡 Попробуй меньшее значение (например, 0.01 или 0.005)"
            )
        else:
            # Отправляем найденные аннотированные изображения
            for file_path in saved_files[:5]:  # Ограничиваем до 5 файлов, чтобы не спамить
                if os.path.exists(file_path):
                    file_name = Path(file_path).name
                    await message.answer_photo(
                        photo=FSInputFile(file_path),
                        caption=f"📷 {file_name}"
                    )

            # Отправляем статистику
            if stats and stats["total_objects"] > 0:
                stats_text = "<b>📊 Статистика анализа:</b>\n"
                stats_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
                stats_text += f"📁 Обработано кадров: {stats['frames_processed']}\n"
                stats_text += f"✨ Всего объектов: {stats['total_objects']}\n\n"

                stats_text += "<b>🔭 Обнаруженные объекты:</b>\n"
                for typ, cnt in stats["by_type"].items():
                    # Эмодзи для разных типов объектов
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
            elif stats:
                await message.answer(
                    "🔍 Объекты не найдены с таким порогом.\n"
                    "💡 Попробуй уменьшить порог для более чувствительного поиска."
                )
            else:
                await message.answer(" Анализ завершён.")

    except Exception as e:
        await message.answer(
            f" Произошла ошибка:\n<code>{str(e)}</code>\n\n"
            f"Попробуй:\n"
            f"• Отправить другое фото\n"
            f"• Изменить порог\n"
            f"• Связаться с разработчиком",
            parse_mode="HTML"
        )
        print(f"Ошибка при анализе: {e}")

    finally:
        # Удаляем временный файл
        if os.path.exists(photo_path):
            try:
                os.unlink(photo_path)
            except Exception as e:
                print(f"Не удалось удалить {photo_path}: {e}")

        # Сбрасываем состояние
        await state.clear()