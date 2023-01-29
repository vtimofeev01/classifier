from flask_cors import CORS
from w3classifier.app import app, dbs
import yaml

def app_run(cfg):
    # print('settings:')
    # print(cfg.pretty())
    print(cfg)
    dbs.load(path=cfg['data']["dataset"], persons_reidentificator=cfg['dnn']['reidentificator'])
    print(dbs.main.columns)
    CORS(app)
    app.debug = True
    app.run(host=cfg['addr']['host'], port=cfg['addr']['port'])


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        app_run(yaml.load(f, Loader=yaml.loader.SafeLoader))
