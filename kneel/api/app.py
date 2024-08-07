"""
    This micro-service takes a dicom image in and returns JSON with localized landmark coordinates.
    (c) Aleksei Tiulpin, University of Oulu, 2019
"""
import argparse
import pathlib
from flask import jsonify
from flask import Flask, request
from gevent.pywsgi import WSGIServer
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
import logging

import base64

from .pipeline import KneeAnnotatorPipeline

app = Flask(__name__)


@app.route('/kneel/predict/bilateral', methods=['POST'])
def analyze_knee():
    logger = logging.getLogger(f'kneel-backend:app')
    logger.info('Received DICOM')
    dicom_base64 = request.get_json(force=True)['dicom']
    dicom_binary = base64.b64decode(dicom_base64)
    raw = DicomBytesIO(dicom_binary)
    data = dcmread(raw)
    logger.info('DICOM read')
    landmarks = annotator.predict(data, args.roi_size_mm, args.pad, args.refine).squeeze()
    logger.info('Prediction successful')
    if landmarks is not None:
        res = {'R': landmarks[0].tolist(), 'L': landmarks[1].tolist(), }
    else:
        res = {'R': None, 'L': None}
    logger.info('Sending results back to the user')
    return jsonify(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_commit', default='bf5f5f2b5f9703f0b3ae6ff973bcf2650c553980', help='Hugging Face commit')
    parser.add_argument('--hf_cache', default='/tmp/', help="Cache directory for the models")
    parser.add_argument('--hf_token', default="", help="Hugging Face token")
    parser.add_argument('--roi_size_mm', type=int, default=140)
    parser.add_argument('--pad', type=int, default=300)
    parser.add_argument('--device',  default='cuda')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--addr', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--deploy', action='store_true')
    parser.add_argument('--jit_trace', action='store_true')
    parser.add_argument('--logs', default='/tmp/kneel.log', type=pathlib.Path)
    args = parser.parse_args()

    args.logs.parent.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.logs),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(f'kneel-backend:app')

    if args.hf_token == "":
        logger.log(logging.WARNING, 'No Hugging Face token provided. We will use the one you have authorized in the CLI')
        args.hf_token = True
    annotator = KneeAnnotatorPipeline(args.hf_commit, args.hf_cache, args.hf_token, args.device, jit_trace=args.jit_trace)

    if args.deploy:
        http_server = WSGIServer((args.addr, args.port), app, log=logger)
        logger.log(logging.INFO, f'Production server is running @ {args.addr}:{args.port}')
        http_server.serve_forever()
    else:
        logger.log(logging.INFO, f'Debug server is running @ {args.addr}:{args.port}')
        app.run(host=args.addr, port=args.port, debug=True)
