"""One JSON object per log line (used by ``apps/observer/logging.json`` through uvicorn --log-config)."""
import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(record.created)) + f'.{int(record.msecs):03d}',
                 'level': record.levelname, 'logger': record.name, 'msg': record.getMessage()}
        entry.update(getattr(record, 'fields', {}))
        if record.exc_info:
            entry['exc'] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)
