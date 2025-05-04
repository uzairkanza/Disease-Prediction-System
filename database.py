import sqlite3
import os
import pandas as pd
from datetime import datetime
import pytz
import streamlit as st
import threading

class Database:
    def __init__(self, db_path='prediction_data.db'):
        """Initialize database connection with proper path handling"""
        self.db_path = os.path.abspath(db_path)
        self.local_tz = pytz.timezone('Asia/Kolkata')  # Set your timezone
        self._local = threading.local()
        self.initialize_db()
    
    def get_connection(self):
        """Get a thread-local database connection with datetime support"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=10,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            # Register datetime converters
            sqlite3.register_adapter(datetime, self._adapt_datetime)
        return self._local.connection
    
    def _adapt_datetime(self, dt):
        """Convert datetime to ISO format for storage"""
        if dt.tzinfo is not None:
            dt = dt.astimezone(pytz.UTC)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _convert_datetime(self, timestamp):
        """Convert stored timestamp to local timezone"""
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return timestamp.replace(tzinfo=pytz.UTC).astimezone(self.local_tz)
    
    def close_connection(self):
        """Close the thread-local database connection"""
        if hasattr(self._local, 'connection'):
            try:
                self._local.connection.close()
            except:
                pass
            del self._local.connection
    
    def initialize_db(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
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
            prediction_date TIMESTAMP
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
            prediction_date TIMESTAMP
        )
        ''')
        conn.commit()
    
    def save_diabetes_prediction(self, user_data, prediction):
        """Save diabetes prediction with current local time"""
        try:
            prediction_time = datetime.now(self.local_tz)
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO diabetes_predictions 
            (name, sex, email, pregnancies, glucose, blood_pressure, skin_thickness, 
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
                prediction_time
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            st.error(f"Error saving diabetes prediction: {str(e)}")
            conn.rollback()
            raise
    
    def save_heart_disease_prediction(self, user_data, prediction):
        """Save heart disease prediction with current local time"""
        try:
            prediction_time = datetime.now(self.local_tz)
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
                prediction_time
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            st.error(f"Error saving heart disease prediction: {str(e)}")
            conn.rollback()
            raise
    
    def save_heart_disease_prediction(self, user_data, prediction):
        """Save heart disease prediction to database with fresh timestamp"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Use database's CURRENT_TIMESTAMP for consistency
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
        except Exception as e:
            st.error(f"Error saving heart disease prediction: {str(e)}")
            conn.rollback()
            raise
    
    def get_all_diabetes_predictions(self):
        """Get all diabetes predictions with proper datetime handling"""
        conn = self.get_connection()
        query = """
        SELECT *, 
               strftime('%Y-%m-%d %H:%M:%S', prediction_date) as formatted_date 
        FROM diabetes_predictions 
        ORDER BY prediction_date DESC
        """
        return pd.read_sql_query(query, conn)
    
    def get_all_heart_disease_predictions(self):
        """Get all heart disease predictions with proper datetime handling"""
        conn = self.get_connection()
        query = """
        SELECT *, 
               strftime('%Y-%m-%d %H:%M:%S', prediction_date) as formatted_date 
        FROM heart_disease_predictions 
        ORDER BY prediction_date DESC
        """
        return pd.read_sql_query(query, conn)
    
    def get_diabetes_predictions_by_email(self, email):
        """Get diabetes predictions for a specific email with fresh data"""
        conn = self.get_connection()
        query = """
        SELECT *, 
               strftime('%Y-%m-%d %H:%M:%S', prediction_date) as formatted_date 
        FROM diabetes_predictions 
        WHERE email = ? 
        ORDER BY prediction_date DESC
        """
        return pd.read_sql_query(query, conn, params=(email,))
    
    def get_heart_disease_predictions_by_email(self, email):
        """Get heart disease predictions for a specific email with fresh data"""
        conn = self.get_connection()
        query = """
        SELECT *, 
               strftime('%Y-%m-%d %H:%M:%S', prediction_date) as formatted_date 
        FROM heart_disease_predictions 
        WHERE email = ? 
        ORDER BY prediction_date DESC
        """
        return pd.read_sql_query(query, conn, params=(email,))

    def refresh_connection(self):
        """Force a refresh of the database connection"""
        self.close_connection()
        return self.get_connection()

# Create a singleton instance with proper path handling
db = Database()
