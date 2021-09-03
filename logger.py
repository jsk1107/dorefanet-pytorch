import logging
import os
import json
import logging.config


def get_logger(save_dir, log_config='./logging.json', level=logging.INFO):

    if not os.path.exists(log_config):
        logging.basicConfig(level=level)
        raise FileExistsError(f'{log_config} 파일을 찾을 수 없습니다.')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(log_config, 'rt', encoding='utf-8') as f:
        cfg = json.load(f)

        # 저장위치 변경
        for _, handler in cfg['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = f'{save_dir}/{handler["filename"]}'
        logging.config.dictConfig(cfg)
        logger = logging.getLogger()

    return logger

