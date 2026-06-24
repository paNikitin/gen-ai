"""
Генератор заявок на курсы повышения квалификации (ДПО).
Использует LLM и стратификацию по городам, специальностям, курсам.
"""
import csv
import json
import random
import time
from collections import Counter
from typing import List
from llm_client import make_client, get_model
from schema import Application, CITIES, SPECIALITIES, DESIRED_COURSES
from dotenv import load_dotenv

load_dotenv()

client = make_client()
MODEL = get_model()
N_APPLICATIONS = 50

# Few-shot примеры (разнообразие, а не шаблон)
FEW_SHOT_EXAMPLES = [
    {
        "full_name": "Кузнецова Елена Владимировна",
        "age": 29,
        "address": {"city": "Волгоград", "district": "Центральный"},
        "speciality": "Маркетолог",
        "desired_course": "Бизнес-аналитика",
        "years_of_experience": 6,
        "graduation_year": 2017
    },
    {
        "full_name": "Смирнов Дмитрий Александрович",
        "age": 48,
        "address": {"city": "Новосибирск", "district": "Академгородок"},
        "speciality": "Инженер",
        "desired_course": "Цифровая трансформация бизнеса",
        "years_of_experience": 24,
        "graduation_year": 1998
    },
    {
        "full_name": "Морозова Анна Петровна",
        "age": 31,
        "address": {"city": "Екатеринбург", "district": "Центральный"},
        "speciality": "Программист",
        "desired_course": "Искусственный интеллект в бизнесе",
        "years_of_experience": 8,
        "graduation_year": 2015
    }
]


def generate_one(seed_city: str, seed_speciality: str, seed_course: str) -> Application:
    """Генерирует одну заявку через LLM."""
    example = random.choice(FEW_SHOT_EXAMPLES)
    prompt = f"""
Ты — генератор РАЗНООБРАЗНЫХ учебных заявок на курсы повышения квалификации.

Вот ПРИМЕР правильной заявки (НЕ КОПИРУЙ ЕГО, а создай свою уникальную!):
{json.dumps(example, ensure_ascii=False, indent=2)}

Теперь сгенерируй НОВУЮ, ОТЛИЧАЮЩУЮСЯ заявку со следующими параметрами:
- Город: {seed_city}
- Специальность: {seed_speciality}
- Желаемый курс: {seed_course}

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Придумай РАЗНЫЕ фамилии (не повторяй Иванов, Петров, Сидоров)
2. Используй РАЗНЫЕ имена (мужские и женские)
3. Возраст должен соответствовать опыту и году окончания
4. Район должен быть РЕАЛЬНЫМ для указанного города
5. Каждая новая заявка должна отличаться от предыдущих

Сделай заявку УНИКАЛЬНОЙ и РЕАЛИСТИЧНОЙ!
"""
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "Ты генератор РАЗНООБРАЗНЫХ учебных заявок. Каждая заявка должна быть уникальной. Используй разные фамилии, имена, возраст. Никогда не используй фамилию 'Иванов'!"},
                {"role": "user", "content": prompt}
            ],
            response_model=Application,
            max_retries=3,
            temperature=1.2
        )
    except Exception as e:
        raise e


def save_to_csv(applications: List[Application], filename: str = "applications.csv"):
    """Сохраняет список заявок в CSV."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'full_name', 'age', 'city', 'district', 'speciality',
            'desired_course', 'years_of_experience', 'graduation_year'
        ])
        for app in applications:
            writer.writerow([
                app.full_name, app.age, app.address.city, app.address.district,
                app.speciality, app.desired_course, app.years_of_experience,
                app.graduation_year
            ])


def plot_distributions(applications: List[Application]):
    """Создаёт и сохраняет гистограммы городов и специальностей."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    city_counts = Counter(app.address.city for app in applications)
    spec_counts = Counter(app.speciality for app in applications)
    last_names = [app.full_name.split()[0] if app.full_name else '?' for app in applications]
    name_counts = Counter(last_names)
    ages = [app.age for app in applications]
    total = len(applications)

    # Города (cities.png)
    plt.figure(figsize=(12, 6))
    cities_sorted = dict(sorted(city_counts.items(), key=lambda x: x[1], reverse=True))
    plt.bar(cities_sorted.keys(), cities_sorted.values(), color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.title('Распределение заявок по городам', fontsize=14, fontweight='bold')
    plt.ylabel('Количество заявок', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    for i, (city, count) in enumerate(cities_sorted.items()):
        plt.text(i, count + 0.3, f'{count/total*100:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('cities.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Специальности (specialities.png)
    plt.figure(figsize=(12, 6))
    specs_sorted = dict(sorted(spec_counts.items(), key=lambda x: x[1], reverse=True))
    plt.bar(specs_sorted.keys(), specs_sorted.values(), color='lightcoral', edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.title('Распределение заявок по специальностям', fontsize=14, fontweight='bold')
    plt.ylabel('Количество заявок', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    for i, (spec, count) in enumerate(specs_sorted.items()):
        plt.text(i, count + 0.3, f'{count/total*100:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('specialities.png', dpi=150, bbox_inches='tight')
    plt.close()

    return name_counts, ages


def check_criteria(applications: List[Application]):
    """Проверяет выполнение критерия 'хорошо'."""
    total = len(applications)
    city_counts = Counter(app.address.city for app in applications)
    spec_counts = Counter(app.speciality for app in applications)

    max_city_pct = max(city_counts.values()) / total * 100 if city_counts else 0
    max_spec_pct = max(spec_counts.values()) / total * 100 if spec_counts else 0

    print("\n" + "="*60)
    print("ПРОВЕРКА КРИТЕРИЯ 'ХОРОШО'")
    print("="*60)
    print(f"  По городам: {max_city_pct:.1f}% {'✅' if max_city_pct <= 40 else '❌'} (должно быть ≤40%)")
    print(f"  По специальностям: {max_spec_pct:.1f}% {'✅' if max_spec_pct <= 35 else '❌'} (должно быть ≤35%)")
    print(f"  Валидных заявок: {total}/50 {'✅' if total == 50 else '❌'}")

    if max_city_pct <= 40 and max_spec_pct <= 35 and total == 50:
        print("\n  🎉 КРИТЕРИЙ 'ХОРОШО' ВЫПОЛНЕН!")
        return True
    else:
        print("\n  ⚠️ КРИТЕРИЙ 'ХОРОШО' НЕ ВЫПОЛНЕН")
        return False


def print_statistics(applications: List[Application], name_counts: Counter, ages: List[int]):
    """Выводит статистику в консоль."""
    total = len(applications)
    city_counts = Counter(app.address.city for app in applications)
    spec_counts = Counter(app.speciality for app in applications)

    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"  Уникальных фамилий: {len(name_counts)} из {total}")
    print(f"  Диапазон возрастов: {min(ages)}-{max(ages)} лет")
    print(f"  Средний возраст: {sum(ages)/len(ages):.1f} лет")
    print(f"\n  Распределение по городам:")
    for city, count in sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {city}: {count} ({count/total*100:.1f}%)")
    print(f"\n  Распределение по специальностям:")
    for spec, count in sorted(spec_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {spec}: {count} ({count/total*100:.1f}%)")


def main():
    print("=" * 70)
    print("ГЕНЕРАТОР ЗАЯВОК НА КУРСЫ ДПО")
    print("=" * 70)
    print(f"LLM сама генерирует имена, фамилии, возраст")
    print(f"Температура: 1.2 | Few-shot: {len(FEW_SHOT_EXAMPLES)} примеров")
    print("Стратификация: города, специальности, курсы")
    print("=" * 70)

    applications = []
    cities_list = list(CITIES)
    specialities_list = [s for s in SPECIALITIES.__args__]
    courses_list = [c for c in DESIRED_COURSES.__args__]

    for i in range(N_APPLICATIONS):
        seed_city = cities_list[i % len(cities_list)]
        seed_speciality = specialities_list[i % len(specialities_list)]
        seed_course = courses_list[i % len(courses_list)]

        print(f"[{i+1:2d}/{N_APPLICATIONS}] {seed_city:20s} | {seed_speciality:15s} | {seed_course[:25]:25s}", end=" ")

        try:
            app = generate_one(seed_city, seed_speciality, seed_course)
            applications.append(app)

            last_name = app.full_name.split()[0] if app.full_name else "?"
            district_short = app.address.district[:15] if app.address.district else "?"
            print(f"✓ {last_name:15s} | {app.age:2d} лет | {district_short}")
        except Exception as e:
            print(f"✗ Ошибка: {str(e)[:50]}")
            time.sleep(0.5)

    print(f"\n{'=' * 70}")
    print(f"Успешно сгенерировано: {len(applications)} из {N_APPLICATIONS}")

    # Сохраняем результаты
    save_to_csv(applications)

    # Строим гистограммы
    print("\nСоздание гистограмм...")
    name_counts, ages = plot_distributions(applications)

    # Проверяем критерии
    criteria_met = check_criteria(applications)

    # Выводим статистику
    print_statistics(applications, name_counts, ages)

    print("\n" + "=" * 70)
    print("✅ ГОТОВО! Созданы файлы:")
    print("  • applications.csv - 50 заявок")
    print("  • cities.png - гистограмма по городам")
    print("  • specialities.png - гистограмма по специальностям")
    print("\n📝 НЕ ЗАБУДЬТЕ:")
    print("  • Написать выводы.md от себя (пример ниже)")
    print("=" * 70)


if __name__ == "__main__":
    main()