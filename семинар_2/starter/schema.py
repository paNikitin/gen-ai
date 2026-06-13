"""
Pydantic-схема для заявки на курсы повышения квалификации (ДПО).
Обновлённые списки городов, специальностей и курсов.
"""
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, field_validator

# ---------- Обновлённые списки ----------
CITIES = {
    "Волгоград", "Воронеж", "Краснодар", "Омск",
    "Томск", "Иркутск", "Хабаровск", "Ярославль",
    "Тюмень", "Калининград", "Саратов", "Тверь"
}

SPECIALITIES = Literal[
    "Архитектор", "Дизайнер", "Логист", "Маркетолог",
    "Психолог", "Социолог", "Финансист", "Эколог",
    "IT-специалист", "Переводчик"
]

DESIRED_COURSES = Literal[
    "Управление персоналом",
    "Бизнес-аналитика",
    "Корпоративная этика",
    "Инновационные технологии",
    "Развитие soft skills",
    "Тайм-менеджмент",
    "Основы криптографии"
]


class Address(BaseModel):
    city: str = Field(description="Город проживания")
    district: str = Field(description="Район или округ города")
    
    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    full_name: str = Field(description="Полное имя")
    age: int = Field(ge=22, le=65, description="Возраст от 22 до 65")
    address: Address
    speciality: SPECIALITIES = Field(description="Текущая специальность")
    desired_course: DESIRED_COURSES = Field(description="Желаемый курс повышения квалификации")
    years_of_experience: int = Field(ge=0, le=40, description="Лет опыта")
    graduation_year: int = Field(ge=1980, le=2024, description="Год окончания вуза")
    
    @field_validator("graduation_year")
    @classmethod
    def validate_graduation_consistency(cls, v: int, info) -> int:
        """Проверяет, что возраст и год окончания не противоречат друг другу."""
        current_year = date.today().year
        age = info.data.get('age')
        if age is not None:
            age_at_graduation = v - (current_year - age)
            if age_at_graduation < 20 or age_at_graduation > 30:
                raise ValueError(
                    f'Несоответствие: возраст {age} и год окончания {v}. '
                    f'Выпускник не может быть младше 20 лет в год окончания'
                )
        return v