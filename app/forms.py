from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class SearchForm(FlaskForm):
    linker = StringField("linker")
    metal = StringField("metal")
    submit = SubmitField("Search")
