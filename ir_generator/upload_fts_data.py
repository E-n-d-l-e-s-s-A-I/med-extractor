import json
import sys
import os
import pyodbc


def upload_fts_data(fts_data_file, config_file):
    # Чтение базы FTS из файла
    with open(fts_data_file, "r", encoding="utf-8") as file:
        fts_data = json.load(file)

    # Чтение конфигурации
    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    config_db_connection = config["db_connection"]
    config_server = config_db_connection["server"]
    config_database = config_db_connection["database"]
    config_username = config_db_connection["username"]
    config_password = config_db_connection["password"]
    config_terminology_base_id = config["terminology_base_id"]

    connectionString = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={config_server};DATABASE={config_database};UID={config_username};PWD={config_password};Trusted_Connection=yes;TrustServerCertificate=yes;"

    conn = pyodbc.connect(connectionString)
    cursor = conn.cursor()

    SQL_STATEMENT = """
    INSERT INTO [dbo].[TermValues]
            ([TerminologyBaseId]
            ,[TermGroupName]
            ,[TermName]
            ,[FeatureName]
            ,[ValueName]
            ,[TermKeywords]
            ,[FeatureKeywords]
            ,[ValueKeywords]
            ,[TermPath]
            ,[FeaturePath]
            ,[ValuePath])
        VALUES
            (?
            ,?
            ,?
            ,?
            ,?
            ,?
            ,?
            ,?
            ,?
            ,?
            ,?)
    """

    for data in fts_data:
        cursor.execute(
            SQL_STATEMENT,
            config_terminology_base_id,
            data["term_group_name"],
            data["term_name"],
            data["feature_name"],
            data["value_name"],
            data["term_keywords"],
            data["feature_keywords"],
            data["value_keywords"],
            data["term_path"],
            data["feature_path"],
            data["value_path"],
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: upload_fts_data.py fts_data.json config.json")
        sys.exit(1)

    fts_data_file = sys.argv[1]
    config_file = sys.argv[2]

    if not os.path.exists(fts_data_file):
        print(f"Knowledge base file {fts_data_file} does not exist")
        sys.exit(1)

    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist")
        sys.exit(1)

    upload_fts_data(fts_data_file, config_file)
