import sqlite3
import os
import pandas as pd
from datetime import datetime
import threading

class Database:
    def __init__(self, db_path='prediction_data.db'):
        self.db_path = db_path
        self._local = threading.local()
        self.initialize_db()

    def get_connection(self):
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def close_connection(self):
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection

    def initialize_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create schema_version table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )
        ''')
        conn.commit()

        current_version = self.get_schema_version()
        if current_version < 1:
            self._migrate_to_v1()
        if current_version < 2:
            self._migrate_to_v2()

    def get_schema_version(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else 0

    def _set_schema_version(self, version):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        conn.commit()

    def _migrate_to_v1(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diabetes_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            sex TEXT,
            email TEXT,
            pregnancies INTEGER,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            diabetes_pedigree REAL,
            age INTEGER,
            prediction TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS heart_disease_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            age INTEGER,
            sex TEXT,
            chest_pain_type TEXT,
            resting_bp REAL,
            cholesterol REAL,
            fasting_bs TEXT,
            resting_ecg TEXT,
            max_heart_rate INTEGER,
            exercise_angina TEXT,
            oldpeak REAL,
            st_slope TEXT,
            major_vessels INTEGER,
            thalassemia TEXT,
            prediction TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        self._set_schema_version(1)

    def _migrate_to_v2(self):
        # Example: In future migrations, you can add new columns or indexes here
        self._set_schema_version(2)

    def save_diabetes_prediction(self, user_data, prediction):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO diabetes_predictions 
        (name, sex, email, pregnancies, glucose, blood_pressure, skin_thickness, 
         insulin, bmi, diabetes_pedigree, age, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_data.get('name', ''),
            user_data.get('sex', ''),
            user_data.get('email', ''),
            user_data.get('pregnancies', 0),
            user_data.get('glucose', 0),
            user_data.get('blood_pressure', 0),
            user_data.get('skin_thickness', 0),
            user_data.get('insulin', 0),
            user_data.get('bmi', 0),
            user_data.get('diabetes_pedigree', 0),
            user_data.get('age', 0),
            prediction
        ))

        conn.commit()
        return cursor.lastrowid

    def save_heart_disease_prediction(self, user_data, prediction):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO heart_disease_predictions 
        (name, email, age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
         resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope, major_vessels, 
         thalassemia, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_data.get('name', ''),
            user_data.get('email', ''),
            user_data.get('age', 0),
            user_data.get('sex', ''),
            user_data.get('chest_pain_type', ''),
            user_data.get('resting_bp', 0),
            user_data.get('cholesterol', 0),
            user_data.get('fasting_bs', ''),
            user_data.get('resting_ecg', ''),
            user_data.get('max_heart_rate', 0),
            user_data.get('exercise_angina', ''),
            user_data.get('oldpeak', 0),
            user_data.get('st_slope', ''),
            user_data.get('major_vessels', 0),
            user_data.get('thalassemia', ''),
            prediction
        ))

        conn.commit()
        return cursor.lastrowid

    def get_all_diabetes_predictions(self):
        conn = self.get_connection()
        query = "SELECT * FROM diabetes_predictions ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn)

    def get_all_heart_disease_predictions(self):
        conn = self.get_connection()
        query = "SELECT * FROM heart_disease_predictions ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn)

    def get_diabetes_prediction_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT prediction, COUNT(*) as count 
        FROM diabetes_predictions 
        GROUP BY prediction
        ''')
        results = cursor.fetchall()
        return {result[0]: result[1] for result in results}

    def get_heart_disease_prediction_stats(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT prediction, COUNT(*) as count 
        FROM heart_disease_predictions 
        GROUP BY prediction
        ''')
        results = cursor.fetchall()
        return {result[0]: result[1] for result in results}

    def get_diabetes_predictions_by_email(self, email):
        conn = self.get_connection()
        query = "SELECT * FROM diabetes_predictions WHERE email = ? ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn, params=(email,))

    def get_heart_disease_predictions_by_email(self, email):
        conn = self.get_connection()
        query = "SELECT * FROM heart_disease_predictions WHERE email = ? ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn, params=(email,))

# Singleton instance
db = Database()
