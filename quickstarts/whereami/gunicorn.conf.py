import multiprocessing
import os
#from logging.config import dictConfig

#loglevl = 'debug'
#disable_redirect_access_to_syslog = True
#accesslog = '-'
#accesslog = 'access.log'
#access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

'''dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})'''

host_ip = os.getenv("HOST", "0.0.0.0")
host=host_ip.strip('[]') # stripping out the brackets if present
port=os.environ.get('PORT', 8080)

bind = host + ":" + str(port)
workers = multiprocessing.cpu_count() * 2 + 1