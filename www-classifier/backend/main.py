import hydra
from flask_cors import CORS

from .app import app, dbs


@hydra.main(config_path='config.yaml', strict=False)
def app_run(cfg):
    print('settings:')
    print(cfg.pretty())
    dbs.load(path=cfg.data.dataset)
    CORS(app.app)
    app.run(host=cfg.addr.host, port=cfg.addr.port, debug=cfg.debug)


if __name__ == '__main__':
    app_run()
