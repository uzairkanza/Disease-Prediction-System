import sqlite3
import os
import pandas as pd
from datetime import datetime
import threading

class Database:
    def __init__(self, db_path='prediction_data.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self._local = threading.local()
        self.initialize_db()
    
    def get_connection(self):
        """Get a thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
        return self._local.connection
    
    def close_connection(self):
        """Close the thread-local database connection"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection
    
    def initialize_db(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create table for diabetes predictions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS diabetes_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
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
            prediction_date 
        )
        ''')
        
        # Create table for heart disease predictions
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
            prediction_date 
        )
        ''')
        
        conn.commit()
    
    def save_diabetes_prediction(self, user_data, prediction):
        """Save diabetes prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO diabetes_predictions 
        (name,sex, email, pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, diabetes_pedigree, age, prediction, prediction_date)
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
            prediction,
            datetime.now().replace(microsecond=0)
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def save_heart_disease_prediction(self, user_data, prediction):
        """Save heart disease prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO heart_disease_predictions 
        (name, email, age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, 
        resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope, major_vessels, 
        thalassemia, prediction, prediction_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            prediction,
            datetime.now().replace(microsecond=0)
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_all_diabetes_predictions(self):
        """Get all diabetes predictions from database"""
        conn = self.get_connection()
        query = "SELECT * FROM diabetes_predictions ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn)
    
    def get_all_heart_disease_predictions(self):
        """Get all heart disease predictions from database"""
        conn = self.get_connection()
        query = "SELECT * FROM heart_disease_predictions ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn)
    
    def get_diabetes_prediction_stats(self):
        """Get statistics for diabetes predictions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get the count of each prediction type
        cursor.execute('''
        SELECT prediction, COUNT(*) as count 
        FROM diabetes_predictions 
        GROUP BY prediction
        ''')
        
        results = cursor.fetchall()
        return {result[0]: result[1] for result in results}
    
    def get_heart_disease_prediction_stats(self):
        """Get statistics for heart disease predictions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get the count of each prediction type
        cursor.execute('''
        SELECT prediction, COUNT(*) as count 
        FROM heart_disease_predictions 
        GROUP BY prediction
        ''')
        
        results = cursor.fetchall()
        return {result[0]: result[1] for result in results}
    
    def get_diabetes_predictions_by_email(self, email):
        """Get diabetes predictions for a specific email"""
        conn = self.get_connection()
        query = "SELECT * FROM diabetes_predictions WHERE email = ? ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn, params=(email,))
    
    def get_heart_disease_predictions_by_email(self, email):
        """Get heart disease predictions for a specific email"""
        conn = self.get_connection()
        query = "SELECT * FROM heart_disease_predictions WHERE email = ? ORDER BY prediction_date DESC"
        return pd.read_sql_query(query, conn, params=(email,))

# Create a singleton instance
db = Database() 
