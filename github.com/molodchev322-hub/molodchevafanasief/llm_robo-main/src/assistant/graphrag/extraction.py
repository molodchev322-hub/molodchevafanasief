"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from __future__ import annotations

import json
import re
from typing import Any

from assistant.infra.config import vocabulary
from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)


def _extract_valid_codes(text: str) -> set[str]:
    """Извлекает все валидные коды из текста (с учетом русских букв и пробелов)."""
    # Ищем паттерн: 1-3 буквы, пробел или дефис, 2-4 цифры, опционально еще дефис и цифры, 
    # опционально буква в конце.
    pattern = r'(?<![A-ZА-Я])([A-ZА-Я]{1,3})[\s\-]+(\d{2,4})([\s\-]+\d{2,4})?([A-ZА-Я]?)(?![A-ZА-Я])'
    matches = re.findall(pattern, text.upper())
    
    # Таблица замены русских букв на английские
    trans = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
    codes = set()
    for prefix, digits, sub_digits, suffix in matches:
        p = prefix.translate(trans)
        s = suffix.translate(trans)
        d2 = sub_digits.translate(trans) if sub_digits else ""
        codes.add(f"{p}-{digits}{d2}{s}")
    return codes


def _build_messages(chunk: Chunk, vocab: dict[str, list[str]]) -> list[dict]:
    schema_description = (
        "Сущности: " + ", ".join(ENTITY_TYPES) + ".\n"
        "Связи и направление (subject -> object):\n"
        + "\n".join(f"{name}: {src} -> {dst}" for name, (src, dst) in RELATION_SIGNATURES.items())
    )
    
    example = (
        "Пример. Текст: «В модели AR-210 при ошибке E-17 отказывает пусковое реле P-1042, "
        "деталь поставляет КомпрессорМаш.»\n"
        "Ответ:\n"
        "{\n"
        '  "entities": [\n'
        '    {"type": "Model", "name": "AR-210"},\n'
        '    {"type": "ErrorCode", "name": "E-17"},\n'
        '    {"type": "Component", "name": "пусковое реле"},\n'
        '    {"type": "Part", "name": "P-1042"},\n'
        '    {"type": "Supplier", "name": "КомпрессорМаш"}\n'
        "  ],\n"
        '  "relations": [\n'
        '    {"subject_type": "ErrorCode", "subject": "E-17", "type": "INDICATES", \n'
        '     "object_type": "Component", "object": "пусковое реле"},\n'
        '    {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", \n'
        '     "object_type": "Component", "object": "пусковое реле"},\n'
        '    {"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", \n'
        '     "object_type": "Part", "object": "P-1042"},\n'
        '    {"subject_type": "Part", "subject": "P-1042", "type": "SUPPLIED_BY", \n'
        '     "object_type": "Supplier", "object": "КомпрессорМаш"}\n'
        "  ]\n"
        "}"
    )

    system = (
        "Ты — эксперт по извлечению знаний из технической документации.\n"
        "Возвращай строго валидный JSON по заданной схеме.\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО переводить слова на английский язык. "
        "Используй ТОЛЬКО те слова, которые есть в тексте чанка.\n"
        "2. Строго соблюдай направление связей (subject -> object) согласно схеме. "
        "Например, связь SUPPLIED_BY всегда идет от Part к Supplier, а HAS_PART от Model к Part.\n"
        "3. Для связи кода ошибки (ErrorCode) с компонентом используй тип INDICATES.\n"
        "4. Извлекай только сущности и связи, прямо подтверждённые текущим чанком. "
        "Не додумывай промежуточные узлы и не переноси факты из других документов.\n"
        "5. Каждую уникальную сущность и каждую уникальную связь возвращай ровно один раз. "
        "Если направление или тип связи неясны, пропусти её вместо догадки.\n"
        "6. Не добавляй пояснений вне JSON и не повторяй элементы для увеличения ответа.\n"
        f"{schema_description}\n"
        "Допустимые названия (ProductLine, Component, Supplier) — только из словаря:\n"
        f"{json.dumps(vocab, ensure_ascii=False, indent=2)}\n"
        f"{example}"
    )
    user = f"Текст чанка:\n\"\"\"\n{chunk.text}\n\"\"\"\nВерни JSON:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_entity(
    raw: dict, chunk_text: str, vocab: dict[str, list[str]]
) -> tuple[Entity | None, str | None]:
    etype = raw.get("type")
    if etype not in ENTITY_TYPES:
        return None, f"неизвестный тип сущности {etype!r}"
    
    name = str(raw.get("name", "")).strip()
    if not name:
        return None, "пустое имя"

    if etype in ("Model", "Part", "ErrorCode", "Procedure"):
        name_upper = name.upper().replace(" ", "-")
        trans = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
        norm = name_upper.translate(trans)
        
        valid_codes = _extract_valid_codes(chunk_text)
        if norm not in valid_codes:
            return None, f"код {name!r} отсутствует в тексте чанка"
            
        return Entity(type=etype, name=norm), None
    else:
        allowed_names = vocab.get(etype, [])
        normalized_name = (
            name.lower()
            .replace("ооо", "")
            .replace("«", "")
            .replace("»", "")
            .replace('"', "")
            .strip()
        )

        canon = None
        for allowed in allowed_names:
            normalized_allowed = (
                allowed.lower()
                .replace("ооо", "")
                .replace("«", "")
                .replace("»", "")
                .replace('"', "")
                .strip()
            )
            if normalized_allowed == normalized_name:
                canon = allowed
                break

        if canon is None:
            return None, f"название {name!r} не из словаря"

        return Entity(type=etype, name=canon), None


def _is_valid_response(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("entities"), list) or not isinstance(parsed.get("relations"), list):
        return False
    for ent in parsed["entities"]:
        if not isinstance(ent, dict) or "type" not in ent or "name" not in ent:
            return False
    for rel in parsed["relations"]:
        if not isinstance(rel, dict):
            return False
        if not all(k in rel for k in ("subject", "subject_type", "type", "object", "object_type")):
            return False
    return True

def safe_json_loads(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Пытаемся "починить" обрезанный JSON
        for suffix in ["}", "]}", "}}", "}]}"]:
            try:
                return json.loads(raw_text + suffix)
            except json.JSONDecodeError:
                continue
        # ВАЖНО: Если не смогли починить, вызываем ошибку!
        raise

def _extract_explicit_facts(
    chunk: Chunk, vocab: dict[str, list[str]]
) -> tuple[list[Entity], list[Relation]]:
    """Извлекает однозначные структурированные факты из шаблонов документов.

    Это не замена LLM-извлечению: функция дополняет его только там, где связь
    явно записана в корпусе (строки состава узлов, перечень кодов ошибок и
    стандартный формат сервисного бюллетеня). Такой слой не позволяет модели
    случайно потерять критичную связь при генерации неполного JSON.
    """
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}

    def add_relation(
        subject_type: str,
        subject_name: str,
        relation_type: str,
        object_type: str,
        object_name: str,
    ) -> None:
        subject, subject_error = _validate_entity(
            {"type": subject_type, "name": subject_name}, chunk.text, vocab
        )
        obj, object_error = _validate_entity(
            {"type": object_type, "name": object_name}, chunk.text, vocab
        )
        if subject_error or object_error or subject is None or obj is None:
            return
        relation = Relation(subject=subject, type=relation_type, object=obj)
        if not relation.matches_schema():
            return
        entities[subject.node_id] = subject
        entities[obj.node_id] = obj
        relations[(subject.node_id, relation_type, obj.node_id)] = relation

    lines = chunk.text.splitlines()

    # Заголовок руководства однозначно задаёт модель. Заголовок повторяется в
    # каждом чанке функцией chunk_document, поэтому контекст раздела не теряется.
    model_name: str | None = None
    if chunk.doc_id.startswith("manual_") and lines:
        model_codes = [code for code in find_codes(lines[0]) if code_type(code) == "Model"]
        if len(model_codes) == 1:
            model_name = model_codes[0]
            model, error = _validate_entity(
                {"type": "Model", "name": model_name}, chunk.text, vocab
            )
            if not error and model is not None:
                entities[model.node_id] = model

    # Модель -> линейка: извлекаем только из прямой формулировки «входит в линейку».
    if model_name:
        for line in lines:
            if "входит в линейку" not in line.casefold():
                continue
            product_lines = [name for kind, name in find_names(line, vocab) if kind == "ProductLine"]
            if len(product_lines) == 1:
                add_relation("Model", model_name, "BELONGS_TO", "ProductLine", product_lines[0])

    # Строки состава узлов имеют фиксированную структуру:
    # «- название узла – деталь P-...., поставщик ...».
    if chunk.doc_id.startswith("manual_"):
        for line in lines:
            if not line.lstrip().startswith(("-", "*")) or "деталь" not in line.casefold():
                continue
            part_codes = [code for code in find_codes(line) if code_type(code) == "Part"]
            components = [name for kind, name in find_names(line, vocab) if kind == "Component"]
            suppliers = [name for kind, name in find_names(line, vocab) if kind == "Supplier"]
            if len(part_codes) != 1 or len(components) != 1:
                continue
            part_name, component_name = part_codes[0], components[0]
            add_relation("Part", part_name, "IS_A", "Component", component_name)
            if model_name:
                add_relation("Model", model_name, "HAS_PART", "Part", part_name)
            if len(suppliers) == 1:
                add_relation("Part", part_name, "SUPPLIED_BY", "Supplier", suppliers[0])

    # Строки перечня ошибок: «E-.. — ..., отказ узла «название»».
    for line in lines:
        if "отказ узла" not in line.casefold():
            continue
        error_codes = [code for code in find_codes(line) if code_type(code) == "ErrorCode"]
        components = [name for kind, name in find_names(line, vocab) if kind == "Component"]
        if len(error_codes) == 1 and len(components) == 1:
            add_relation("ErrorCode", error_codes[0], "INDICATES", "Component", components[0])

    # Регламент явно указывает код ошибки, для которого он применяется.
    if chunk.doc_id.startswith("procedure_"):
        for line in lines:
            if "применяется при ошибке" not in line.casefold():
                continue
            codes = find_codes(line)
            error_codes = [code for code in codes if code_type(code) == "ErrorCode"]
            procedure_codes = [code for code in codes if code_type(code) == "Procedure"]
            if len(error_codes) == 1 and len(procedure_codes) == 1:
                add_relation("ErrorCode", error_codes[0], "FIXED_BY", "Procedure", procedure_codes[0])

    # В бюллетенях замена описана явно: старая деталь снята с производства,
    # вместо неё устанавливается новая. Порядок кодов в абзаце задаёт направление.
    if chunk.doc_id.startswith("bulletin_"):
        for paragraph in re.split(r"\n\s*\n", chunk.text):
            lowered = paragraph.casefold()
            if "снята с производства" not in lowered or "вместо" not in lowered:
                continue
            part_codes = [code for code in find_codes(paragraph) if code_type(code) == "Part"]
            if len(part_codes) < 2:
                continue
            old_part, new_part = part_codes[0], part_codes[1]
            add_relation("Part", old_part, "REPLACED_BY", "Part", new_part)
            components = [name for kind, name in find_names(paragraph, vocab) if kind == "Component"]
            if len(components) == 1:
                add_relation("Part", old_part, "IS_A", "Component", components[0])

        for line in lines:
            if "поставщик" not in line.casefold():
                continue
            part_codes = [code for code in find_codes(line) if code_type(code) == "Part"]
            suppliers = [name for kind, name in find_names(line, vocab) if kind == "Supplier"]
            if len(part_codes) == 1 and len(suppliers) == 1:
                add_relation("Part", part_codes[0], "SUPPLIED_BY", "Supplier", suppliers[0])

    return list(entities.values()), list(relations.values())


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    vocab = vocabulary()
    messages = _build_messages(chunk, vocab)

    raw = None
    for _attempt in range(2):
        try:
            resp = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
            print("\n=== СЫРОЙ ОТВЕТ ОТ НЕЙРОСЕТИ ===\n", resp.content, "\n=================================\n")
            
            # Иногда модели оборачивают JSON в теги ```json ... ```. Убираем их.
            content_clean = resp.content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
                
            parsed = safe_json_loads(content_clean.strip())

            # === НАЧАЛО УМНОГО ФИЛЬТРА ===
            if parsed and "relations" in parsed:
                # Словарь перевода английских галлюцинаций модели на русский
                EN_TO_RU = {
                    "compressor": "компрессор", "condenser_fan": "вентилятор конденсатора",
                    "temperature_sensor": "датчик температуры", "controller": "контроллер",
                    "pressure_relais": "реле давления", "pressure_relay": "реле давления",
                    "evaporator_fan": "вентилятор испарителя", "lighting_block": "блок освещения",
                    "dehumidifier": "осушитель", "thermostatic_valve": "терморегулирующий вентиль",
                    "start_relay": "пусковое реле", "flow_relay": "реле протока",
                    "circulation_pump": "циркуляционный насос"
                }

                for rel in parsed["relations"]:
                    # 1. Переводим английские сущности на русский
                    for key in ["subject", "object"]:
                        val = str(rel.get(key, "")).lower().strip()
                        if val in EN_TO_RU:
                            rel[key] = EN_TO_RU[val]
                            
                    # 2. Разворачиваем SUPPLIED_BY (чтобы всегда было Part -> Supplier)
                    if rel.get("type") == "SUPPLIED_BY" and rel.get("subject_type") == "Supplier":
                        rel["subject"], rel["object"] = rel["object"], rel["subject"]
                        rel["subject_type"], rel["object_type"] = rel["object_type"], rel["subject_type"]

                    # 3. Исправляем HAS_PART на INDICATES для кодов ошибок
                    if rel.get("type") == "HAS_PART" and rel.get("subject_type") == "ErrorCode":
                        rel["type"] = "INDICATES"
                        
                    # 4. Исправляем HAS_PART на BELONGS_TO для линеек
                    if (rel.get("type") == "HAS_PART" and 
                        rel.get("subject_type") == "ProductLine" and 
                        rel.get("object_type") == "ProductLine"):
                        rel["type"] = "BELONGS_TO"

                # Обновляем сущности в списке entities, чтобы они тоже стали русскими
                if "entities" in parsed:
                    for e in parsed["entities"]:
                        val = str(e.get("name", "")).lower().strip()
                        if val in EN_TO_RU:
                            e["name"] = EN_TO_RU[val]
            # === КОНЕЦ УМНОГО ФИЛЬТРА ===
            
            if _is_valid_response(parsed):
                raw = parsed
                break
            else:
                print("\n!!! ОШИБКА ВАЛИДАЦИИ: _is_valid_response вернул False !!!\n")
        except Exception as e:
            # ТЕПЕРЬ МЫ УВИДИМ ОШИБКУ
            print(f"\n!!! ОШИБКА ПРИ ЗАПРОСЕ ИЛИ ПАРСИНГЕ: {type(e).__name__}: {e} !!!\n")
        
        raw = None

    if raw is None:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, dropped=1)

    dropped = 0
    entities: dict[str, Entity] = {}
    by_name: dict[str, Entity] = {}

    for ent_raw in raw["entities"]:
        ent, err = _validate_entity(ent_raw, chunk.text, vocab)
        if err:
            dropped += 1
            continue
        entities[ent.node_id] = ent
        # Сохраняем ключи строго в нижнем регистре
        by_name[str(ent_raw.get("name", "")).strip().lower()] = ent
        by_name[ent.name.lower()] = ent

    relations: list[Relation] = []
    for rel_raw in raw["relations"]:
        rtype = rel_raw.get("type")
        if rtype not in RELATION_SIGNATURES:
            dropped += 1
            continue
            
        subj_name = str(rel_raw.get("subject", "")).strip()
        obj_name = str(rel_raw.get("object", "")).strip()
        
        if subj_name.lower() in by_name:
            subj = by_name[subj_name.lower()] # Берем по нижнему регистру
        else:
            subj_type = rel_raw.get("subject_type")
            subj, err = _validate_entity({"type": subj_type, "name": subj_name}, chunk.text, vocab)
            if err:
                dropped += 1
                continue
            entities[subj.node_id] = subj
            by_name[subj_name.lower()] = subj
            
        if obj_name.lower() in by_name:
            obj = by_name[obj_name.lower()] # Берем по нижнему регистру
        else:
            obj_type = rel_raw.get("object_type")
            obj, err = _validate_entity({"type": obj_type, "name": obj_name}, chunk.text, vocab)
            if err:
                dropped += 1
                continue
            entities[obj.node_id] = obj
            by_name[obj_name.lower()] = obj
            
        rel = Relation(subject=subj, type=rtype, object=obj)
        if not rel.matches_schema():
            dropped += 1
            continue
        relations.append(rel)

    # Дополняем LLM-ответ фактами из однозначных шаблонов документов.
    explicit_entities, explicit_relations = _extract_explicit_facts(chunk, vocab)
    for entity in explicit_entities:
        entities.setdefault(entity.node_id, entity)

    explicit_keys = {
        (rel.subject.node_id, rel.type, rel.object.node_id) for rel in explicit_relations
    }
    authoritative_subjects = {(subject, relation_type) for subject, relation_type, _ in explicit_keys}
    validated_relations: list[Relation] = []
    for relation in relations:
        key = (relation.subject.node_id, relation.type, relation.object.node_id)
        if (key[0], key[1]) in authoritative_subjects and key not in explicit_keys:
            # Для связи, прямо заданной шаблонной строкой источника, не принимаем
            # альтернативный объект, предложенный моделью без текстового основания.
            dropped += 1
            continue
        validated_relations.append(relation)

    relation_map = {
        (rel.subject.node_id, rel.type, rel.object.node_id): rel for rel in validated_relations
    }
    for relation in explicit_relations:
        relation_map.setdefault(
            (relation.subject.node_id, relation.type, relation.object.node_id), relation
        )

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=list(entities.values()),
        relations=list(relation_map.values()),
        dropped=dropped,
    )