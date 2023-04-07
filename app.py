from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length
import sqlite3
from sqlite3 import Error

def create_connection():
    conn = None;
    try:
        conn = sqlite3.connect('users.db')
        print("Connection to SQLite DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return conn

def create_table(conn):
    try:
        query = '''CREATE TABLE IF NOT EXISTS users 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                email TEXT NOT NULL, 
                password TEXT NOT NULL)'''
        conn.execute(query)
        print("Table created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

conn = create_connection()
if conn is not None:
    create_table(conn)
    conn.close()
else:
    print("Connection to SQLite DB failed")



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-goes-here'

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        conn = create_connection()
        if conn is not None:
            query = f"INSERT INTO users (email, password) VALUES (?, ?)"
            params = (email, password)
            conn.execute(query, params)
            conn.commit()
            conn.close()
            return redirect(url_for('home'))
        else:
            print("Connection to SQLite DB failed")
    return render_template('login.html', form=form)


@app.route('/home')
def home():
    return 'Welcome to the home page!'

if __name__ == '__main__':
    app.run(debug=True)
