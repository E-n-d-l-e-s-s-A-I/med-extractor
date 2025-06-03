import json
import sys
import os
import datetime
from ir_generator import utilities
# import pyodbc


def generate_ir(input_file, template_file, output_file, fts_db_config_file):
    with open(input_file, "r", encoding="utf-8") as file:
        input = json.load(file)

    with open(template_file, "r", encoding="utf-8") as file:
        template = json.load(file)

    # Чтение конфигурации
    with open(fts_db_config_file, "r", encoding="utf-8") as file:
        fts_db_config = json.load(file)

    config_db_connection = fts_db_config["db_connection"]
    config_server = config_db_connection["server"]
    config_database = config_db_connection["database"]
    config_username = config_db_connection["username"]
    config_password = config_db_connection["password"]
    config_terminology_base_id = fts_db_config["terminology_base_id"]

    connectionString = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={config_server};DATABASE={config_database};UID={config_username};PWD={config_password};Trusted_Connection=yes;TrustServerCertificate=yes;"

    # conn = pyodbc.connect(connectionString)
    # db_cursor = conn.cursor()
    db_cursor = None

    log = []
    analyze_node(template, input, db_cursor, config_terminology_base_id, log)

    # conn.commit()
    # conn.close()

    remove_if_empty(template)
    utilities.save_as_json(template, output_file)
    utilities.save_as_json(log, "ir_generator_log.json")


def remove_if_empty(node):
    if "successors" in node:
        successors = node["successors"]
        if len(successors) == 0:
            if "template" not in node:
                return
            if "remove_if_empty" in node and not node["remove_if_empty"]:
                return

            node["remove"] = True
            return

        items_to_remove = []
        for s in successors:
            remove_if_empty(s)
            if "remove" in s and s["remove"]:
                items_to_remove.append(s)
        for item in items_to_remove:
            successors.remove(item)


# TODO: Исправить ошибку при добавлении характеристик одного и того же термина
# Пример: при импорте инфоресурса из файла возникла ошибка (12.11.2024-12:02:21): ru.dvo.iacp.is.iacpaas.storage.exceptions.StorageException:
# ru.dvo.iacp.is.iacpaas.storage.generator.exceptions.StorageGenerateException: Невозможно выполнить порождение
# от понятия 'Кожные покровы (нетерминал)' по метаотношению '[Выбор типа значений (нетерминал) → + 'set' all Характеристика (нетерминал)]',
# так как последнее уже не может быть использовано для порождения (поскольку единственно-возможное порождение уже выполнено)
def analyze_node(node, input, db_cursor, tb_id, log):
    if "timestamp" in node and node["timestamp"]:
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S:%f")
        node["name"] = node["name"] + " " + date_time

    if "template" not in node:
        if "successors" in node:
            for s in node["successors"]:
                analyze_node(s, input, db_cursor, tb_id, log)
        return

    template = node["template"]
    template_type = template["type"]

    if template_type == "Группа признаков" or template_type == "Факты":
        process_feature_group(input, node)
    elif template_type == "Ссылки":
        process_links(input, node, db_cursor, tb_id, log)


def process_links(input, node, db_cursor, tb_id, log):
    template = node["template"]
    template_block_path = template["block_path"]
    term_group_names = template["term_group_names"]
    group_name = template["node_meta"] if "node_meta" in template else "Признак"
    name = node["name"]

    successors = []
    node["successors"] = successors

    blocks = find_text_block(input, template_block_path)

    for block in blocks:
        if block is None or "data" not in block:
            continue

        data = block["data"]
        kb_query_data = []
        for d in data:
            if "Характеристики" in d:
                if "Имя" in d:
                    # Обработка особого случая когда у элемента такое же имя как и у раздела
                    if d["Имя"] == name:
                        c_data = d["Характеристики"]
                        for c_d in c_data:
                            if "Имя" in c_d and "Значение" in c_d:
                                kb_query_data.append(
                                    {
                                        "term_name": c_d["Имя"],
                                        "feature_name": None,
                                        "value_name": c_d["Значение"],
                                    }
                                )
                    # Остальные варианты
                    else:
                        for c in d["Характеристики"]:
                            if "Имя" in c and "Значение" in c:
                                kb_query_data.append(
                                    {
                                        "term_name": d["Имя"],
                                        "feature_name": c["Имя"],
                                        "value_name": c["Значение"],
                                    }
                                )

            elif "Имя" in d and "Значение" in d:
                kb_query_data.append(
                    {
                        "term_name": d["Имя"],
                        "feature_name": None,
                        "value_name": d["Значение"],
                    }
                )

        SQL_STATEMENT = """
        DECLARE @T_FT_QUERY NVARCHAR(200)
        DECLARE @F_FT_QUERY NVARCHAR(200)
        DECLARE @V_FT_QUERY NVARCHAR(200)
        DECLARE @TERMINOLOGY_BASE_ID INT
        DECLARE @TERM_GROUP_NAMES NVARCHAR(200)
        --DECLARE @TGN table (N NVARCHAR(200))

        SET @T_FT_QUERY = ?
        SET @F_FT_QUERY = ?
        SET @V_FT_QUERY = ?
        SET @TERMINOLOGY_BASE_ID = ?
        SET @TERM_GROUP_NAMES = ?
        --insert into @TGN values (SELECT value FROM STRING_SPLIT(@TERM_GROUP_NAMES, ','))

        SELECT TOP 5 KEY_TBL_T.RANK AS T_RANK, KEY_TBL_F.RANK AS F_RANK, KEY_TBL_V.RANK AS V_RANK, FT_TBL.*
        FROM [TermValues] AS FT_TBL 
            INNER JOIN  
            FREETEXTTABLE ([TermValues], (TermKeywords), @T_FT_QUERY) AS KEY_TBL_T
            ON FT_TBL.Id = KEY_TBL_T.[KEY]
            INNER JOIN
            FREETEXTTABLE ([TermValues], (FeatureKeywords), @F_FT_QUERY) AS KEY_TBL_F
            ON KEY_TBL_T.[KEY] = KEY_TBL_F.[KEY]	 
            INNER JOIN
            FREETEXTTABLE ([TermValues], (ValueKeywords), @V_FT_QUERY) AS KEY_TBL_V
            ON KEY_TBL_F.[KEY] = KEY_TBL_V.[KEY]
        WHERE [TerminologyBaseId] = @TERMINOLOGY_BASE_ID AND [TermGroupName] IN (SELECT value FROM STRING_SPLIT(@TERM_GROUP_NAMES, ','))

        UNION

        SELECT TOP 5 KEY_TBL_T.RANK AS T_RANK, 0 AS F_RANK, KEY_TBL_V.RANK AS V_RANK, FT_TBL.*
        FROM [TermValues] AS FT_TBL 
            INNER JOIN  
            FREETEXTTABLE ([TermValues], (TermKeywords), @T_FT_QUERY) AS KEY_TBL_T
            ON FT_TBL.Id = KEY_TBL_T.[KEY]
            INNER JOIN
            FREETEXTTABLE ([TermValues], (ValueKeywords), @V_FT_QUERY) AS KEY_TBL_V
            ON KEY_TBL_T.[KEY] = KEY_TBL_V.[KEY]
        WHERE [TerminologyBaseId] = @TERMINOLOGY_BASE_ID AND [TermGroupName] IN (SELECT value FROM STRING_SPLIT(@TERM_GROUP_NAMES, ','))

        ORDER BY T_RANK DESC, F_RANK DESC, V_RANK DESC
        """

        print("Querying KB database")
        for d in kb_query_data:
            term_name = d["term_name"].strip(" ")
            feature_name = d["feature_name"]
            feature_name = feature_name.strip(" ") if feature_name else None
            value_name = d["value_name"].strip(" ")

            # Замена слов являющихся стоп-словами для FTS системы на аналоги
            if value_name.casefold() == "нет".casefold():
                value_name = "отсутствует"
            if value_name.casefold() == "да".casefold():
                value_name = "имеется"

            if not term_name or not value_name:
                continue

            # TODO я убрал
            # db_cursor.execute(
            #     SQL_STATEMENT,
            #     term_name,
            #     feature_name if feature_name else "наличие присутствие",
            #     value_name,
            #     tb_id,
            #     ",".join(term_group_names)
            # )

            # result = db_cursor.fetchall()
            result = ""
            if len(result) > 0:
                r = result[0]
                d["kb_term_name"] = r[6]
                d["kb_feature_name"] = r[7]
                d["kb_value_name"] = r[8]
                d["term_path"] = r[12]
                d["feature_path"] = r[13]
                d["value_path"] = r[14]

        # db_cursor.commit()
        print("Querying completed")

        for d in kb_query_data:
            if "term_path" not in d:
                continue
            term_path = d["term_path"]

            if "feature_path" in d and d["feature_path"] is not None:
                cq = get_existing_list_item(successors, "name", d["kb_term_name"])
                if cq is None:
                    cq = {}
                    successors.append(cq)
                    cq["name"] = d["kb_term_name"]
                    cq["type"] = "НЕТЕРМИНАЛ"
                    cq["meta"] = group_name

                if "successors" in cq:
                    cq_successors = cq["successors"]
                else:
                    cq_successors = []
                    cq["successors"] = cq_successors

                set_original(cq, term_path)

                append_qualitative_value(
                    cq_successors,
                    d["kb_feature_name"],
                    "Характеристика",
                    d["kb_value_name"],
                    None,
                    None,
                    None,
                    d["feature_path"],
                    d["value_path"],
                )

            else:
                append_qualitative_value(
                    successors,
                    d["kb_term_name"],
                    group_name,
                    d["kb_value_name"],
                    None,
                    None,
                    None,
                    term_path,
                    d["value_path"],
                )

        log.extend(kb_query_data)


# TODO обработка случая когда в исходном файле с данными для одного и того же термина есть значения просто и значения через характеристики
# Пример [{"имя": "имя", "значение": "значения"}, {"имя": "имя", "Характеристики": [{...}, {...}, {...}]}]
def process_feature_group(input, node):
    template = node["template"]
    template_type = template["type"]
    template_block_path = template["block_path"]
    name = node["name"]

    successors = []
    node["successors"] = successors

    if template_type == "Группа признаков":
        group_name = "Признак"
    elif template_type == "Факты":
        group_name = "Факт"
    else:
        return

    blocks = find_text_block(input, template_block_path)

    for block in blocks:
        if block is None or "data" not in block:
            continue

        data = block["data"]
        for d in data:
            d_name = d["Имя"] if "Имя" in d else None
            if not d_name:
                continue

            if "Характеристики" in d:
                # Обработка особого случая когда у элемента такое же имя как и у раздела
                if d_name == name:
                    c_data = d["Характеристики"]
                    for c_d in c_data:
                        append_qualitative_value(
                            successors,
                            c_d["Имя"],
                            group_name,
                            c_d["Значение"] if "Значение" in c_d else None,
                            c_d["Единица"] if "Единица" in c_d else None,
                            c_d["Дата"] if "Дата" in c_d else None,
                            c_d["Время"] if "Время" in c_d else None,
                            None,
                            None,
                        )
                # Остальные варианты
                else:
                    cq = get_existing_list_item(successors, "name", d_name)
                    if cq is None:
                        cq = {}
                        successors.append(cq)

                        cq["name"] = d_name
                        cq["type"] = "НЕТЕРМИНАЛ"
                        cq["meta"] = group_name

                    if "successors" in cq:
                        cq_successors = cq["successors"]
                    else:
                        cq_successors = []
                        cq["successors"] = cq_successors

                    for c in d["Характеристики"]:
                        if not isinstance(c, dict):
                            continue
                        append_qualitative_value(
                            cq_successors,
                            c["Имя"],
                            "Характеристика",
                            c["Значение"] if "Значение" in c else None,
                            c["Единица"] if "Единица" in c else None,
                            c["Дата"] if "Дата" in c else None,
                            c["Время"] if "Время" in c else None,
                            None,
                            None,
                        )

            else:
                if "Имя" in d:
                    append_qualitative_value(
                        successors,
                        d_name,
                        group_name,
                        d["Значение"] if "Значение" in d else None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )


def get_existing_list_item(list, key, value):
    existing_item = None
    for item in list:
        if key in item and item[key] == value:
            existing_item = item
            break
    return existing_item


def set_original(node, value):
    if value is None:
        return
    if "original" in node:
        node["warning"] = 'Ссылка "original" была перезаписана'
    node["original"] = value


def append_qualitative_value(
    parent_successors, name, meta, value, measure, date, time, feature_path, value_path
):
    v = get_existing_list_item(parent_successors, "name", name)
    if v is None:
        v = {}
        parent_successors.append(v)

        # {
        #     "id" : 16054587752744,
        #     "name" : "Слабость",
        #     "type" : "НЕТЕРМИНАЛ",
        #     "meta" : "Признак",
        #     "successors" :
        #     [

        #     {
        #     "id" : 16054587752748,
        #     "name" : "Качественные значения",
        #     "type" : "НЕТЕРМИНАЛ",
        #     "meta" : "Качественные значения",
        #     "successors" :
        #     [

        #     {
        #         "id" : 16054587752754,
        #         "value" : "Присутствует",
        #         "type" : "ТЕРМИНАЛ-ЗНАЧЕНИЕ",
        #         "valtype" : "STRING",
        #         "meta" : "значение"
        #     }                ]
        #     }              ]
        # }

        v["name"] = name
        v["type"] = "НЕТЕРМИНАЛ"
        v["meta"] = meta

    # Значения с пустым именем не добавляются
    if not name:
        return

    set_original(v, feature_path)

    if "successors" in v:
        successors = v["successors"]
    else:
        successors = []
        v["successors"] = successors

    nt = get_existing_list_item(successors, "name", "Качественные значения")
    if nt is None:
        nt = dict()
        successors.append(nt)

        nt["name"] = "Качественные значения"
        nt["type"] = "НЕТЕРМИНАЛ"
        nt["meta"] = "Качественные значения"

    if "successors" in nt:
        nt_successors = nt["successors"]
    else:
        nt_successors = []
        nt["successors"] = nt_successors

    if value is None:
        return v

    nt_val = dict()
    nt_successors.append(nt_val)

    nt_val["type"] = "ТЕРМИНАЛ-ЗНАЧЕНИЕ"
    nt_val["valtype"] = "STRING"
    nt_val["meta"] = "значение"

    set_original(nt_val, value_path)

    value_text = object_to_text(value)

    if measure is not None:
        value_text = value_text + " " + measure

    if date is not None:
        data_list = list()
        if "Год" in date:
            data_list.append(date["Год"])
        if "Месяц" in date:
            data_list.append(date["Месяц"])
        if "День" in date:
            data_list.append(date["День"])
        value_text = value_text + " " + ".".join(data_list)

    if time is not None:
        time_list = list()
        if "Час" in time:
            time_list.append(time["Час"])
        if "Мин" in time:
            time_list.append(time["Мин"])
        if "Сек" in time:
            time_list.append(time["Сек"])
        value_text = value_text + " " + ":".join(time_list)

        if "Период" in time:
            value_text = value_text + " " + time["Период"]

    nt_val["value"] = value_text

    return v


# def find_text_block(input, path):
#     for b in input["blocks"]:
#         result = find_text_block_by_index(b, path, 0)
#         if result is not None:
#             return result
#     return None

# def find_text_block_by_index(input, path, path_index):
#     if input["title"] != path[path_index]:
#         return None
#     elif path_index >= len(path) - 1:
#         return input
#     elif "blocks" in input:
#         for b in input["blocks"]:
#             result = find_text_block_by_index(b, path, path_index + 1)
#             if result is not None:
#                 return result
#         return None
#     else:
#         return None


def object_to_text(d):
    if isinstance(d, dict):
        values = list()
        for k in d:
            values.append(k + ":" + str(d[k]))
        return " ".join(values)
    else:
        return str(d)


def find_text_block(input, path):
    block_list = list()
    for b in input["blocks"]:
        blocks = find_text_block_by_index(b, path, 0)
        block_list.extend(blocks)
    return block_list


def find_text_block_by_index(input, path, path_index):
    block_list = list()

    if input["title"] != path[path_index]:
        return block_list
    elif path_index >= len(path) - 1:
        block_list.append(input)
        return block_list
    elif "blocks" in input:
        for b in input["blocks"]:
            block = find_text_block_by_index(b, path, path_index + 1)
            block_list.extend(block)
        return block_list
    else:
        return block_list


# generate_ir("АкопянСГ_history_0.json", "ИБ - Шаблон.json", "АкопянСГ_history_0_ir.json", "fts_db_config.json")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: ir_generator.py input.json template.json output.json fts_db_config.json"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    template_file = sys.argv[2]
    output_file = sys.argv[3]
    fts_db_config_file = sys.argv[4]

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(1)

    if not os.path.exists(template_file):
        print(f"Template file {template_file} does not exist")
        sys.exit(1)

    if not os.path.exists(fts_db_config_file):
        print(f"FTS DB config file {fts_db_config_file} does not exist")
        sys.exit(1)

    generate_ir(input_file, template_file, output_file, fts_db_config_file)
