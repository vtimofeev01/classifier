import hydra
from flask_cors import CORS
from w3classifier.app import app, dbs


@hydra.main(config_path='config.yaml', strict=False)
def app_run(cfg):
    print('settings:')
    print(cfg.pretty())
    dbs.load(path=cfg.data.dataset, persons_reidentificator=cfg.dnn.reidentificator)
    CORS(app)
    app.run(host=cfg.addr.host, port=cfg.addr.port, debug=cfg.debug)


if __name__ == '__main__':
    app_run()
