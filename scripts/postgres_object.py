import psycopg2
from config.config import *


class PostgresClient:

    def __init__(self, host="127.0.0.1", port=5432, password=""):
        self.host = host
        self.port = port
        self.password = password
        try:
            self.con = psycopg2.connect(database="postgres", user="postgres", password=password, host=host, port=port)
        except Exception as e:
            print(e)
        self.table_to_columns = {'public': {}}
        print("Database connected successfully")

    def reconnect(self):
        if self.con.closed:
            self.con = psycopg2.connect(database="postgres", user="postgres", password=self.password,
                                        host=self.host, port=self.port)
            print("Reconnect to postgresDB.")

    def close(self):
        if not self.con.closed:
            self.con.close()
            print("Close connection with postgresDB.")

    def _validate_insertion(self, table, insertion_value, scheme="public"):
        table_format = self.get_table_format(table=table, scheme=scheme)
        insert_values_format = {(k, type(v)) for k, v in insertion_value.items()}
        return insert_values_format == table_format

    def _format_value_to_table(self, value):
        if type(value) == str:
            return "'" + value.replace("'", "") + "'"
        elif type(value) == list:
            return str(value).replace("'", "").replace('[', "'{").replace(']', "}'")
        else:
            return str(value).replace("'", "")

    def insert(self, table, insertion_value, scheme="public", allow_update=False):
        if self._validate_insertion(table=table, insertion_value=insertion_value, scheme=scheme):
            query = None
            if not self.is_exists(table=table, insertion_value=insertion_value, scheme=scheme):
                columns = ", ".join(insertion_value.keys())
                values = ', '.join([self._format_value_to_table(value) for value in insertion_value.values()])
                query = f'INSERT INTO {scheme}."{table}"({columns}) VALUES({values});'

            elif allow_update:
                update_set = ', '.join(f"{k} = {self._format_value_to_table(v)}" for k, v in insertion_value.items())
                table_key = self._get_table_key(table=table, scheme=scheme)
                query = f"UPDATE {scheme}.\"{table}\" SET {update_set} WHERE {table_key} = {self._format_value_to_table(insertion_value[table_key])};"

            if query is not None:
                cur = self.con.cursor()
                cur.execute(query)
                self.con.commit()
            else:
                print(f"Couldn't insert: {insertion_value} to {scheme}.{table}. You can override the existing table row using 'allow_update'=True.")
        else:
            print(f"Couldn't insert: {insertion_value} to {scheme}.{table}. insertion doesn't match table format.")

    def is_exists(self, table, insertion_value, scheme="public"):
        table_key = self._get_table_key(table=table, scheme=scheme)
        query = f"select count(*) from {scheme}.\"{table}\" where \
                {table_key}={self._format_value_to_table(insertion_value.get(table_key, ''))}"
        cur = self.con.cursor()
        cur.execute(query)
        return cur.fetchall()[0][0] > 0

    def select(self, table, select="*", condition="", scheme="public", limit="", order_by=""):
        query = f"SELECT {select} FROM {scheme}.\"{table}\""
        if condition:
            query += f" WHERE {condition}\n"
        if order_by:
            query += f"ORDER BY {order_by} DESC\n"
        if limit:
            query += f"LIMIT {limit}\n"

        query += ";"
        cur = self.con.cursor()
        cur.execute(query)
        return cur.fetchall()

    def _get_table_key(self, table, scheme="public"):
        query = f"SELECT \"column_name\" FROM information_schema.key_column_usage where table_name = '{table}' AND table_schema='{scheme}';"
        cur = self.con.cursor()
        cur.execute(query)
        response = cur.fetchall()
        return response[0][0]

    def get_table_format(self, table, scheme="public"):
        if scheme not in self.table_to_columns:
            self.table_to_columns[scheme] = dict()

        if table in self.table_to_columns[scheme]:
            return self.table_to_columns[scheme][table]

        query = f"SELECT * FROM information_schema.columns WHERE table_schema='{scheme}' AND table_name='{table}';"
        cur = self.con.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        table_format = {(row[3], POSTGRES_TYPE_TO_PYTHON[row[7]]) for row in rows}
        self.table_to_columns[scheme][table] = table_format
        return table_format


if __name__ == "__main__":
    postgres_client = PostgresClient(password=DB_PASSWORD)
