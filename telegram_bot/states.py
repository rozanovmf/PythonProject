from aiogram.fsm.state import State, StatesGroup


class PhotoAnalysis(StatesGroup):
    waiting_for_photo = State()
    waiting_for_threshold = State()