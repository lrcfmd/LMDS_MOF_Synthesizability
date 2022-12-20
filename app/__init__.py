from flask import Flask


app = Flask(__name__)
app.config['SECRET_KEY'] = "SOME_SECRET_KEY"


from app import routes
