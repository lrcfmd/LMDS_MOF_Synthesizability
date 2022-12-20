import os
import re
import numpy as np
import joblib
import pandas as pd
from logging.config import dictConfig
from rdkit import Chem
from jinja2 import BaseLoader, TemplateNotFound,ChoiceLoader, FileSystemLoader
from urllib import request, parse
from flask import render_template
from ElMD import ElMD
from mordred import Calculator, descriptors
from app import app
from app.forms import SearchForm

#We split loading of templates between local and remote as common templates
#For LMDS are hosted seperately to reduce code duplication
#Thus we define a quick class to use to fetch these remotely
class UrlLoader(BaseLoader):
    def __init__(self, url_prefix):
        self.url_prefix = url_prefix

    def get_source(self, environment, template):
        url = parse.urljoin(self.url_prefix, template)
        app.logger.debug(url)
        try:
            t = request.urlopen(url)
            if t.getcode() is None or t.getcode() == 200:
                return t.read().decode('utf-8'), None, None
        except IOError:
            pass
        raise TemplateNotFound(template)

app.jinja_loader = ChoiceLoader([app.jinja_loader, UrlLoader("https://lmds.liverpool.ac.uk/static")])

#Set up featurisers and scalers
calc = Calculator(descriptors, ignore_3D=True)
descriptors = ['ABC', 'SpAbs_A', 'SpAD_A', 'nAromAtom', 'nAromBond', 'nC', 'ATS1dv',
       'VR2_DzZ', 'VR2_Dhome/samantha/anaconda3/bin/pythonzm', 'VR2_Dzv', 'VR2_Dzse', 'VR2_Dzpe', 'VR2_Dzare',
       'VR2_Dzp', 'VR2_Dzi', 'BertzCT', 'nBondsO', 'nBondsA', 'nBondsM',
       'nBondsKD', 'C3SP2', 'Xp-1d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d',
       'Xp-7d', 'VR2_Dt', 'VR2_D', 'NaaCH', 'NaasC', 'ETA_beta', 'ETA_beta_ns',
       'ETA_eta_RL', 'ETA_eta_FL', 'PEOE_VSA7', 'SMR_VSA7', 'SlogP_VSA6',
       'MID', 'MID_C', 'MPC2', 'MPC3', 'TpiPC10', 'nRing', 'n6Ring', 'naRing',
       'n6aRing', 'MWC01', 'Zagreb1', 'Zagreb2']
elements = pd.read_csv('app/elemental_descriptors.csv',index_col='Symbol').drop('Name',axis=1)
scaler = joblib.load("app/scaler.joblib")
metal_scaler = joblib.load("app/metal_scaler.joblib")

# Load models
m1 = joblib.load("app/model_M1.joblib")
m2 = joblib.load("app/model_M2.joblib")
m3 = joblib.load("app/model_M3.joblib")

#Configure logging
dictConfig({"version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {
                        "format": '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                }},

            "handlers": {
                "wsgi": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://flask.logging.wsgi_errors_stream",
                    "formatter": "default",
                    }
                },

            "root": {"level": "DEBUG", "handlers": ["wsgi"]},
            })
#Define route
@app.route("/", methods=['GET', 'POST'])
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    form = SearchForm()

    if form.validate_on_submit():
        try:
            #Parse strings
            linkers = [smiles.strip() for smiles in re.split(',|, | ',str(form.linker.data)) \
                          if smiles != ""]
            metals = [metals.strip() for metals in re.split(',|, | ',str(form.metal.data)) \
                          if metals != ""]
            #Load and scale metal descriptors
            metal_descriptors = elements.loc[metals][['Atomic_Number','Atomic_Weight','Atomic Radius','Mulliken EN','polarizability(A^3)','electron affinity(kJ/mol)']].reset_index(drop=True)
            metal_descriptors = pd.DataFrame(metal_scaler.transform(metal_descriptors), columns=metal_descriptors.columns)
            #Load and scale linker descriptors
            results = []
            for linker in linkers:
                mol = Chem.MolFromSmiles(linker)
                results.append(calc(mol).asdict())
            results = pd.DataFrame(results)
            scaled_linker_descriptors = pd.DataFrame(scaler.transform(results), columns=results.columns)[descriptors]
            scaled_linker_descriptors = scaled_linker_descriptors.fillna(0.0)

            #Deal with functionality of matching metals to linkers
            if len(metal_descriptors) == 1:
                metals = metals * len(scaled_linker_descriptors)
                metal_descriptors = pd.concat([metal_descriptors]*len(scaled_linker_descriptors), axis=0)
                metal_descriptors.index = scaled_linker_descriptors.index
            elif len(scaled_linker_descriptors) == 1:
                linkers = linkers * len(metals)
                scaled_linker_descriptors = pd.concat([scaled_linker_descriptors]*len(metal_descriptors), axis=0)
                scaled_linker_descriptors.index = metal_descriptors.index

            #merge metal and linker descriptions
            all_descs = scaled_linker_descriptors.merge(metal_descriptors,  left_index=True, right_index=True)
            # Process predictions
            m1_preds = m1.predict(all_descs)
            m2_preds = m2.predict(all_descs)
            m3_preds = m3.predict(all_descs)
            #Format output
            prediction_texts = []
            for i in range(len(metals)):
                 if not m1_preds[i]:
                     prediction = "porosity < 2.4 Å"
                 elif not m2_preds[i]:
                     prediction = "2.4Å < porosity < 4.4Å"
                 elif not m3_preds[i]:
                     prediction = "4.4Å < porosity <5.9Å"
                 else:
                     prediction = "porosity<5.9 Å"
                 prediction_texts.append(prediction)
            results = zip(linkers, metals, m1_preds, m2_preds, m3_preds, prediction_texts)

            return render_template("MOF_ml.html", form=form, results=results)

        except Exception as e:
            return render_template("MOF_ml.html", form=form, message="Failed to process input, check it is properly formatted")
            app.logger.debug(e)

    return render_template("MOF_ml.html", form=form)
