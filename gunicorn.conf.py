"""One process is required while browser games live in memory."""
import os

bind = '0.0.0.0:' + os.environ.get('PORT', '10000')
workers = 1
worker_class = 'gthread'
threads = 4
timeout = 240
graceful_timeout = 30
accesslog = '-'
errorlog = '-'
capture_output = True
control_socket_disable = True
