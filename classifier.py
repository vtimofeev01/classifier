import logging
import sys
import argparse

from PyQt5.QtWidgets import QApplication
from src.app import CApp

LOGGER = logging.getLogger(__name__)
LOGGERS = [
    LOGGER,
    logging.getLogger('src.app'),
    logging.getLogger('src.view')
]

def argparser():
    parser = argparse.ArgumentParser('Binary Classifier building with PyQt5')
    parser.add_argument('--img-dir', dest='imgdir', required=True)
    parser.add_argument('--l', dest='label')
    parser.add_argument('--lv', dest='labels_values', default='list of label values')
    return parser

def main(args):
    LOGGER.info(args)
    app = QApplication(sys.argv)
    classifier = CApp(args.imgdir, args.label, args.labels_values)
    app.exec()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    parser = argparser()
    main(parser.parse_args())