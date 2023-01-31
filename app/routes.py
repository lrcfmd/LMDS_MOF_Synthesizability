import os
import re
import numpy as np
import joblib
import pandas as pd
from logging.config import dictConfig
from rdkit import Chem
import rdkit
from rdkit.Chem import AllChem
from jinja2 import BaseLoader, TemplateNotFound,ChoiceLoader, FileSystemLoader
from urllib import request, parse
from flask import render_template
from ElMD import ElMD
from mordred import Calculator, descriptors
from app import app
import torch
from app.forms import SearchForm
from app.Encoder import deepSVDD, build_network, build_autoencoder
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


device = 'cpu'
def score(deep_SVDD, X):
    with torch.no_grad():
        net = deep_SVDD.net.to(device)
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        c, R = torch.FloatTensor([deep_SVDD.c]).to(device), torch.FloatTensor([deep_SVDD.R]).to(device)
        dist = torch.sum((y - c)**2, dim=1)
        if deep_SVDD.objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
    return scores

lunar_scaler = joblib.load("app/lunar_scaler.joblib")
lof_scaler = joblib.load("app/lof_scaler.joblib")
deep_scaler = joblib.load("app/deep_scaler.joblib")

# Load models
net_name = 'mof_Net'
clf_deep = deepSVDD.DeepSVDD()
clf_deep.net = build_network(net_name)
clf_deep.ae_net = build_autoencoder(net_name)
clf_deep.net_name = net_name
clf_deep.load_model(model_path='app/deep_model.tar')

clf_lunar = joblib.load('app/clf_LUNAR.joblib')
clf_lof = joblib.load('app/clf_LOF.joblib')

metal_scaled = pd.read_csv('app/metal_scaled.csv')
metal_scaled = metal_scaled.set_index('Symbol')

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
            linker = [smiles.strip() for smiles in re.split(',|, | ',str(form.linker.data)) \
                          if smiles != ""][0]
            metal = [metals.strip() for metals in re.split(',|, | ',str(form.metal.data)) \
                          if metals != ""][0]
            
            app.logger.debug("Linker:")
            app.logger.debug(linker)
            app.logger.debug("Metal:")
            app.logger.debug(metal)
            #Load and scale linker descriptors
            metal_df = metal_scaled.loc[metal,:]
            
            mol = Chem.MolFromSmiles(linker)
            linker_modified = Chem.MolToSmiles(mol)
            fpts = AllChem.GetMorganFingerprintAsBitVect(mol,2,256)
            linker_df = np.array(fpts) # linker features to be used in lof&lunar
            fpts_dl = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
            linker_df_dl = np.array(fpts_dl) # linker features to be used in deep model

            # concatenate metal features & linker features
            df = np.concatenate((metal_df.to_numpy(), linker_df)) # to be used in lof&lunar
            df_dl = np.concatenate((metal_df.to_numpy(), linker_df_dl)) # to be used in deep model
            

            output_lof = clf_lof.decision_function(df.reshape(1,-1))*(-1)
            output_lof = lof_scaler.transform(output_lof.reshape(-1,1)) 
            output_lof = np.round(output_lof[0][0], 3)
            output_lof_predict = output_lof > 0.703

            # lunar model prediction & normalization

            output_lunar = clf_lunar.decision_function(df.reshape(1,-1))*(-1)
            output_lunar = lunar_scaler.transform(output_lunar.reshape(-1,1)) 
            output_lunar = np.round(output_lunar[0][0], 3)
            output_lunar_predict = output_lunar > 0.552

            # deep model prediction & normalization
            output_deep = score(clf_deep, df_dl.reshape(1,-1)).cpu().detach().numpy()*(-1)
            output_deep = deep_scaler.transform(output_deep.reshape(-1,1)) 
            output_deep = np.round(output_deep[0][0], 3)
            output_deep_predict = output_deep > 0.800
            
            # Process predictions
            #Format output
            
            scores = (output_lof, output_lunar, output_deep)
            predictions = (output_lof_predict, output_lunar_predict, output_deep_predict)
                       
            app.logger.debug(scores)
            app.logger.debug(predictions)
            return render_template("MOF_ml.html", form=form, scores=scores, predictions=predictions)

        except Exception as e:
            app.logger.debug(e)
            return render_template("MOF_ml.html", form=form, message="Failed to process input, check it is properly formatted")
            

    return render_template("MOF_ml.html", form=form)
