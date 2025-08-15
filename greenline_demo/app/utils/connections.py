
import os

def get_connection_placeholders():
    return {
        "synapse": {
            "server": os.getenv("SYNAPSE_SERVER", "your-synapse-name.sql.azuresynapse.net"),
            "database": os.getenv("SYNAPSE_DB", "hr_curated"),
            "username": os.getenv("AZURE_SQL_USER", "managed-identity-or-user"),
            "auth": os.getenv("AZURE_SQL_AUTH", "ManagedIdentity | SQLPassword"),
            "driver": os.getenv("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
        },
        "workday": {
            "tenant": os.getenv("WORKDAY_TENANT", "your_tenant"),
            "client_id": os.getenv("WORKDAY_CLIENT_ID", "xxxxxxxx"),
            "scope": os.getenv("WORKDAY_SCOPE", "workday.prism.read"),
            "auth_type": os.getenv("WORKDAY_AUTH", "OAuth2 Client Credentials")
        }
    }

def synapse_connection_example_snippet():
    return '''
# Example (commented): connect to Synapse from Streamlit using pyodbc
# import pyodbc, os
# server = os.getenv("SYNAPSE_SERVER")
# database = os.getenv("SYNAPSE_DB")
# driver = os.getenv("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
# # Managed Identity (on Azure) can use 'Authentication=ActiveDirectoryMsi' via SQLAlchemy/ODBC alternatives.
# conn_str = f"Driver={{ {driver} }};Server=tcp:{server},1433;Database={database};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
# # For SQL auth (dev only): add UID and PWD — do NOT hardcode in code; use environment variables/secrets.
# # conn = pyodbc.connect(conn_str + f";UID={os.getenv('AZURE_SQL_USER')};PWD={os.getenv('AZURE_SQL_PASSWORD')}")
# # cursor = conn.cursor()
# # cursor.execute("SELECT TOP 50 * FROM hr_mart.people_facts")
# # rows = cursor.fetchall()
# # conn.close()
'''
