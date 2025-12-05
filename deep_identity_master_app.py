import os
import json
from datetime import datetime
from typing import Dict, List

import streamlit as st
import pandas as pd
from openai import OpenAI

RESULTS_FILE = "deep_identity_results.json"

POTENTIALS = [
    "Аметист",
    "Гранат",
    "Цитрин",
    "Сапфир",
    "Гелиодор",
    "Изумруд",
    "Янтарь",
    "Рубин",
    "Шунгит",
]
COLUMNS = ["c1", "c2", "c3"]


# ---------- OpenAI клиент ----------

def get_openai_client():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ---------- Работа с результатами ----------

def load_results() -> List[Dict]:
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ---------- Генерация отчёта ----------

def generate_text_report(combined: Dict[str, Dict[str, int]], full_text: str) -> str:
    client = get_openai_client()
    if client is None:
        return (
            "⚠️ OpenAI API ключ не найден (ни в st.secrets, ни в переменной окружения OPENAI_API_KEY).\n"
            "Добавь ключ в настройки, чтобы генерировать отчёты."
        )

    # текстовая таблица для подсказки модели
    lines = []
    for p in POTENTIALS:
        row = combined.get(p, {})
        lines.append(
            f"{p}: c1={row.get('c1', 0)}  c2={row.get('c2', 0)}  c3={row.get('c3', 0)}"
        )
    table_text = "\n".join(lines)

    prompt = f"""
Ты — мастер системы потенциалов (Аметист, Гранат, Цитрин, Сапфир, Гелиодор, Изумруд, Янтарь, Рубин, Шунгит).
У тебя есть итоговая карта 3×3 по столбцам:
- c1: интуиция / восприятие / причина,
- c2: процесс / творчество / проявление,
- c3: результат / инструмент / действие.

Вот числовая карта по потенциалам:

{table_text}

А вот свободные ответы человека по трём блокам (детство, работа, окружение):

\"\"\"{full_text}\"\"\"


Сделай структурированный отчёт:

1. Краткое резюме (3–5 предложений): ядро личности и направление пути.
2. Сильные потенциалы (топ 3–4): как они проявляются и в чём ресурс.
3. Потенциалы, которые пока недоиспользованы, но к ним тянет — куда можно смещать фокус.
4. Возможные смещения и перекосы (аккуратно, без диагнозов).
5. Практические шаги на 4–6 недель: конкретные действия для движения в свою реализацию.

Пиши по-русски, тоном: тёплый, честный, поддерживающий, без эзотерической «воды», но с глубиной.
Опирайся и на цифры, и на текст.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "Ты глубокий, но приземлённый мастер системы потенциалов."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Ошибка при обращении к OpenAI: {e}"


# ---------- UI ----------

def main():
    st.set_page_config(
        page_title="Deep Identity · Мастер-панель",
        layout="wide",
    )

    st.title("Deep Identity — мастер-панель для ассистента")

    results = load_results()
    if not results:
        st.info("Файл deep_identity_results.json пока пустой или не найден. "
                "Сначала пусть клиенты пройдут диагностику в клиентском приложении.")
        return

    # список клиентов
    st.subheader("Список всех прохождений")

    # создаём таблицу для обзора
    rows = []
    for idx, r in enumerate(results):
        combined = r.get("combined", {})
        total_score = sum(
            combined.get(p, {}).get("c1", 0)
            + combined.get(p, {}).get("c2", 0)
            + combined.get(p, {}).get("c3", 0)
            for p in POTENTIALS
        )
        rows.append(
            {
                "№": idx,
                "Дата": r.get("timestamp", "")[:19],
                "Имя": r.get("name", ""),
                "Контакт": r.get("contact", ""),
                "Σ баллов": total_score,
            }
        )

    df_overview = pd.DataFrame(rows)
    st.dataframe(df_overview, use_container_width=True)

    st.markdown("---")
    st.subheader("Работа с конкретным клиентом")

    idx_selected = st.number_input(
        "Выбери № клиента из таблицы выше", min_value=0, max_value=len(results) - 1, step=1, value=0
    )

    entry = results[int(idx_selected)]
    st.markdown(f"**Имя:** {entry.get('name','')}  \n**Контакт:** {entry.get('contact','')}  \n"
                f"**Дата:** {entry.get('timestamp','')[:19]}")

    combined = entry.get("combined", {})
    text = entry.get("text", "")

    # таблица потенциалов
    st.markdown("### Карта 3×3 по потенциалам")

    data = []
    for p in POTENTIALS:
        row = combined.get(p, {})
        data.append(
            {
                "Потенциал": p,
                "c1 (интуиция/восприятие)": row.get("c1", 0),
                "c2 (процесс/проявление)": row.get("c2", 0),
                "c3 (результат/действие)": row.get("c3", 0),
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.markdown(
        "_c1 — как человек воспринимает мир, чувствует причины;  "
        "c2 — как ведёт процесс, проявляет творчество;  "
        "c3 — в каком виде любит выдавать результат и действовать._"
    )

    # текст клиента
    st.markdown("### Свободные ответы клиента")
    st.text_area("Текст", value=text, height=300)

    st.markdown("---")
    if st.button("✨ Сгенерировать AI-отчёт по этому клиенту"):
        with st.spinner("Готовлю отчёт..."):
            report = generate_text_report(combined, text)
        st.subheader("AI-отчёт (черновик для тебя)")
        st.markdown(report)

        st.download_button(
            "💾 Скачать отчёт .txt",
            data=report,
            file_name=f"deep_identity_report_{entry.get('name','client')}.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
