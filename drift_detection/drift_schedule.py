"""
Гибкое расписание работы детекции дрейфа (wall-clock, с таймзоной).

Формат JSON (поле schedule в роуте):

{
  "timezone": "Europe/Moscow",
  "enabled": true,
  "rules": [
    {
      "months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      "month_days": [1, 15, 30],
      "weekdays": [0, 1, 2, 3, 4],
      "time_ranges": [{"start": "09:00", "end": "18:00"}]
    }
  ]
}

Семантика:
- rules объединяются через ИЛИ (достаточно одного совпавшего правила);
- внутри правила поля через И (если поле не задано / пусто — не фильтрует);
- weekdays: 0=пн … 6=вс (ISO), либо mon/tue/…;
- time_ranges: можно через полночь, напр. 22:00–06:00;
- нет schedule / enabled не false / пустой rules → всегда активно.
"""
from __future__ import annotations

import json
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Sequence, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WEEKDAY_ALIASES = {
    "mon": 0, "monday": 0, "пн": 0, "понедельник": 0,
    "tue": 1, "tuesday": 1, "вт": 1, "вторник": 1,
    "wed": 2, "wednesday": 2, "ср": 2, "среда": 2,
    "thu": 3, "thursday": 3, "чт": 3, "четверг": 3,
    "fri": 4, "friday": 4, "пт": 4, "пятница": 4,
    "sat": 5, "saturday": 5, "сб": 5, "суббота": 5,
    "sun": 6, "sunday": 6, "вс": 6, "воскресенье": 6,
}


def parse_schedule(raw: Union[str, dict, None]) -> Optional[Dict[str, Any]]:
    """
    Парсит schedule из JSON-строки или dict.
    None / пусто → None (всегда активно).
    Бросает ValueError при невалидном JSON/схеме.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"schedule: невалидный JSON ({e})") from e
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError("schedule: ожидается JSON-строка или объект")

    if not isinstance(data, dict):
        raise ValueError("schedule: корень должен быть объектом")

    tz_name = str(data.get("timezone") or "Europe/Moscow")
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"schedule: неизвестная timezone '{tz_name}'") from e

    rules = data.get("rules", [])
    if rules is None:
        rules = []
    if not isinstance(rules, list):
        raise ValueError("schedule.rules: ожидается массив")

    normalized_rules: List[Dict[str, Any]] = []
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"schedule.rules[{i}]: ожидается объект")
        normalized_rules.append(_normalize_rule(rule, i))

    return {
        "timezone": tz_name,
        "enabled": bool(data.get("enabled", True)),
        "rules": normalized_rules,
    }


def _normalize_rule(rule: dict, idx: int) -> Dict[str, Any]:
    months = _parse_int_list(rule.get("months"), 1, 12, f"rules[{idx}].months")
    month_days = _parse_int_list(rule.get("month_days"), 1, 31, f"rules[{idx}].month_days")
    weekdays = _parse_weekdays(rule.get("weekdays"), f"rules[{idx}].weekdays")
    time_ranges = _parse_time_ranges(rule.get("time_ranges"), f"rules[{idx}].time_ranges")
    return {
        "months": months,
        "month_days": month_days,
        "weekdays": weekdays,
        "time_ranges": time_ranges,
    }


def _parse_int_list(value: Any, lo: int, hi: int, label: str) -> Optional[List[int]]:
    if value is None or value == []:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"schedule.{label}: ожидается массив чисел")
    out: List[int] = []
    for x in value:
        try:
            n = int(x)
        except Exception as e:
            raise ValueError(f"schedule.{label}: нечисловое значение {x!r}") from e
        if n < lo or n > hi:
            raise ValueError(f"schedule.{label}: {n} вне диапазона {lo}..{hi}")
        out.append(n)
    return sorted(set(out)) or None


def _parse_weekdays(value: Any, label: str) -> Optional[List[int]]:
    if value is None or value == []:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"schedule.{label}: ожидается массив")
    out: List[int] = []
    for x in value:
        if isinstance(x, str):
            key = x.strip().lower()
            if key not in _WEEKDAY_ALIASES:
                raise ValueError(f"schedule.{label}: неизвестный день недели {x!r}")
            out.append(_WEEKDAY_ALIASES[key])
        else:
            try:
                n = int(x)
            except Exception as e:
                raise ValueError(f"schedule.{label}: неверное значение {x!r}") from e
            if n < 0 or n > 6:
                raise ValueError(f"schedule.{label}: weekday {n} вне 0..6 (пн..вс)")
            out.append(n)
    return sorted(set(out)) or None


def _parse_hhmm(s: str, label: str) -> dt_time:
    s = str(s).strip()
    try:
        parts = s.split(":")
        if len(parts) < 2:
            raise ValueError
        h, m = int(parts[0]), int(parts[1])
        sec = int(parts[2]) if len(parts) > 2 else 0
        return dt_time(h, m, sec)
    except Exception as e:
        raise ValueError(f"schedule.{label}: ожидается HH:MM, получено {s!r}") from e


def _parse_time_ranges(value: Any, label: str) -> Optional[List[Dict[str, dt_time]]]:
    if value is None or value == []:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"schedule.{label}: ожидается массив")
    out: List[Dict[str, dt_time]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"schedule.{label}[{i}]: ожидается объект {{start, end}}")
        if "start" not in item or "end" not in item:
            raise ValueError(f"schedule.{label}[{i}]: нужны start и end")
        out.append(
            {
                "start": _parse_hhmm(item["start"], f"{label}[{i}].start"),
                "end": _parse_hhmm(item["end"], f"{label}[{i}].end"),
            }
        )
    return out or None


def _time_in_ranges(t: dt_time, ranges: Sequence[Dict[str, dt_time]]) -> bool:
    for r in ranges:
        start, end = r["start"], r["end"]
        if start <= end:
            if start <= t <= end:
                return True
        else:
            # через полночь: 22:00–06:00
            if t >= start or t <= end:
                return True
    return False


def _rule_matches(rule: Dict[str, Any], dt: datetime) -> bool:
    if rule.get("months") is not None and dt.month not in rule["months"]:
        return False
    if rule.get("month_days") is not None and dt.day not in rule["month_days"]:
        return False
    # ISO: Monday=0 .. Sunday=6
    if rule.get("weekdays") is not None and dt.weekday() not in rule["weekdays"]:
        return False
    if rule.get("time_ranges") is not None:
        if not _time_in_ranges(dt.time().replace(microsecond=0), rule["time_ranges"]):
            return False
    return True


def is_schedule_active(schedule: Optional[Dict[str, Any]], now: Optional[datetime] = None) -> bool:
    """
    True — дрейф сейчас можно считать.
    schedule is None → всегда True.
    """
    if schedule is None:
        return True
    if not schedule.get("enabled", True):
        return False
    rules = schedule.get("rules") or []
    if not rules:
        return True

    tz = ZoneInfo(schedule.get("timezone") or "Europe/Moscow")
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    return any(_rule_matches(rule, now) for rule in rules)
